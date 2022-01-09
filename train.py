import os
import config
import torch
import torch.optim as optim
import time
import numpy as np
import pickle as pkl
from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

from notifier import Notifier

notifier = Notifier(
  #webhook_url='https://discord.com/api/webhooks/796406472459288616/PAkiGGwqe0_PwtBxXYQvOzbk78B4RQP6VWRkvpBtw6Av0sc_mDa3saaIlwVPFjOIeIbt',
  chat_id='293701727',
  api_token='1878628343:AAEFVRsqDz63ycmaLOFS7gvsG969wdAsJ0w'
)

torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, epoch=0):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(epoch=epoch, loss=mean_loss)
    return losses


def main():
    current_time = time.strftime("%Y%m%d_%H%M%S")
    path_looses = f"./loss/{current_time}_all_looses.pkl"
    path_metrics = f"./loss/{current_time}_all_metrics.pkl"
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
      train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
    )

    if config.LOAD_MODEL and os.path.exists(config.CHECKPOINT_FILE):
      load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)
    
    all_looses = []
    metrics_accur = []
    last_mapval = -1

    for epch in range(config.NUM_EPOCHS):
        epoch = epch + 1 + config.START_EPOCH
        #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        looses = train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, epoch)
        looses = np.array(looses)
        all_looses.append({ 'epoch': epoch, 'looses': looses, 'mean': np.mean(looses), 'std': np.std(looses) })
        
        if epoch > 0 and (epoch % 3 == 0 or epoch >= config.NUM_EPOCHS):
            if os.path.exists(path_looses):
                os.remove(path_looses)
            pkl.dump(all_looses, open(path_looses, "wb"))

            metris = check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            ).item()
            metris['map'] = mapval

            txt_metrics = ''
            for m_name, m_value in metris.items():
                txt_metrics += f"{m_name}: {m_value}\n"
            print(txt_metrics)
            metrics_accur.append({ 'epoch': epoch, **metris })

            if os.path.exists(path_metrics):
                os.remove(path_metrics)
            pkl.dump(metrics_accur, open(path_metrics, "wb"))

            if mapval > last_mapval:
                last_mapval = mapval
                save_checkpoint(model, optimizer, filename=config.CHECKPOINT_FILE)

            notifier(title=f'Trainin epoch {epoch}', msg=f'Metrics:\n{txt_metrics}')
            model.train()


if __name__ == "__main__":
    start_time = time.time()
    date_start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    notifier(title='Start training YOLOv3', msg=f'Traning started at {date_start}')
    
    main()
    end_time = time.time()
    date_end = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    total_time = end_time - start_time
    
    print(f"Total time: {total_time}")
    notifier(
        title=f'Finish training YOLOV3',
        msg=f'Start time: {date_start} \nEnd time: {date_end} \nTotal time: {total_time} seconds',
    )
