import pickle , time , base64 , os , sys , pika
import requests
from requests.auth import HTTPBasicAuth
# from Edge import Edge
from src.Cloud import Cloud
from src.Edge import Edge


class Transmission:
    def __init__(self , config):
        self.config = config
        self.model = config["model"]["name"]
        self.batch_size = config["model"]["batch"]
        self.data = config["model"]["data"]

        self.address = config["rabbitmq"]["address"]
        self.username = config["rabbitmq"]["username"]
        self.password = config["rabbitmq"]["password"]
        self.virtual_host = config["rabbitmq"]["virtual-host"]


        credentials = pika.PlainCredentials(
            config["rabbitmq"]["username"], config["rabbitmq"]["password"]
        )
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                config["rabbitmq"]["address"],
                5672,
                config["rabbitmq"]["virtual-host"],
                credentials,
            )
        )
        self.channel = self.connection.channel()

        # queue server_clients
        self.notify_queue = "notify_queue"
        self.notify_stop_queue = "notify_stop_queue"
        self.transmission_queue = "transmission_queue"
        self.channel.queue_declare(queue=self.notify_queue, durable=True)
        self.channel.queue_declare(queue=self.notify_stop_queue, durable=True)
        self.channel.queue_declare(queue=self.transmission_queue , durable=True)

    def push_message(self , queue_name , message):
        self.channel.basic_publish(exchange='',
                                   routing_key=queue_name,
                                   body=pickle.dumps(message)
                                   )

    def listening(self , queue_name ):
        time.sleep(0.05)  # small delay
        method_frame, header_frame, body = self.channel.basic_get(queue=queue_name, auto_ack=True)
        if method_frame and body:
            data = pickle.loads(body)
            # print("[Listening] " , data)
            return data
        else:
            return None


    def server(self):
        # notify START for client2
        num_clients = 2
        for i in range(num_clients):
            self.push_message(self.notify_queue ,
                              {"action" : "START" ,
                                        "info"  : self.config}
                              )

        # wait STOP notify
        stop_counter = 0
        while True :
            message = self.listening(self.notify_stop_queue)
            if message is not None and "action" in message and message["action"] == "STOP":
                stop_counter += 1
            if stop_counter == 2 :
                break

        # delete old queue and exit
        self.delete_old_queues()
        sys.exit(0)



    def edge(self):

        # wait start
        message = None
        while message is None :
            message = self.listening(self.notify_queue)
            if message is not None and message["action"] == "START":
                break

        # get info from config
        config = message["info"]
        my_edge = Edge(config)

        # start inference
        for batch_result in my_edge.run():
            self.push_message(self.transmission_queue , batch_result)

        # send STOP notify
        self.push_message(self.transmission_queue, message={"action": "STOP"})
        self.push_message(self.notify_stop_queue , message= {"action" : "STOP"})

    def cloud(self):

        # wait start
        message = None
        while message is None :
            message = self.listening(self.notify_queue)
            if message is not None and message["action"] == "START":
                break

        # get info from config
        config = message["info"]
        my_cloud = Cloud(config)

        # start inference
        while True :
            data = self.listening(self.transmission_queue)
            if isinstance(data, dict) and "action" in data:
                break

            if data is not None :
                my_cloud.run(data )

        self.push_message(self.notify_stop_queue, message={"action": "STOP"})

    def delete_old_queues(self):
        url = f'http://{self.address}:15672/api/queues'
        response = requests.get(url, auth=HTTPBasicAuth(self.username, self.password))

        if response.status_code == 200:
            queues = response.json()

            credentials = pika.PlainCredentials(self.username, self.password)
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(self.address, 5672, f'{self.virtual_host}', credentials))
            http_channel = connection.channel()

            for queue in queues:
                queue_name = queue['name']
                if queue_name.startswith("reply") or queue_name.startswith(
                        "notify") or queue_name.startswith(
                        "transmission") or queue_name.startswith("rpc_queue"):

                    http_channel.queue_delete(queue=queue_name)

                else:
                    http_channel.queue_purge(queue=queue_name)

            connection.close()
            return True
        else:
            return False



