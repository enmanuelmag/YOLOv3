from notifier import Notifier

nofifier = Notifier(
  webhook_url='https://discord.com/api/webhooks/796406472459288616/PAkiGGwqe0_PwtBxXYQvOzbk78B4RQP6VWRkvpBtw6Av0sc_mDa3saaIlwVPFjOIeIbt',
  chat_id='293701727',
  api_token='1878628343:AAEFVRsqDz63ycmaLOFS7gvsG969wdAsJ0w'
)

nofifier(
  title='Test',
  msg='Testando envio de mensaje'
)