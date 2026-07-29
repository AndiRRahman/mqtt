from __future__ import annotations

import json
import threading
from typing import Any, Callable

import paho.mqtt.client as mqtt

import config


ResultCallback = Callable[
    [dict[str, Any]],
    None
]


class MQTTService:

    def __init__(
        self,
        result_callback: ResultCallback | None = None,
    ) -> None:


        self.result_callback = result_callback


        self._connected = threading.Event()


        self._loop_started = False


        self.client = self._create_client()


    def publish_prediction_command(
        self,
        prediction_result: dict[str, Any]
    ) -> bool:

        if not self.is_connected():
            print(
                "MQTT belum terhubung"
            )
            return False
    
    
        payload = json.dumps(
            {
                "class": prediction_result.get(
                    "label",
                    "Unknown"
                ),
    
                "confidence": prediction_result.get(
                    "confidence",
                    0.0
                ),
    
                "class_id": prediction_result.get(
                    "class_id",
                    -1
                )
            }
        )
    
    
        result = self.client.publish(
            "sampah/command",
            payload,
            qos=1
        )
    
    
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
    
            print(
                "Prediction sent:",
                payload
            )
    
            return True
    
    
        print(
            "Prediction gagal dikirim"
        )
    
        return False
    # ======================================================
    # CREATE MQTT CLIENT
    # ======================================================

    def _create_client(self):


        client = mqtt.Client(
            client_id=config.MQTT_CLIENT_ID
        )


        client.on_connect = (
            self._on_connect
        )


        client.on_disconnect = (
            self._on_disconnect
        )


        client.on_message = (
            self._on_message
        )


        if config.MQTT_USERNAME:


            client.username_pw_set(
                config.MQTT_USERNAME,
                config.MQTT_PASSWORD
            )


        return client



    # ======================================================
    # CONNECT CALLBACK
    # ======================================================


    def _on_connect(
        self,
        client,
        userdata,
        flags,
        rc
    ):


        if rc == 0:


            print(
                "MQTT connected"
            )


            self._connected.set()



            # menerima status ESP32

            client.subscribe(
                "sampah/status"
            )


        else:


            print(
                "MQTT failed:",
                rc
            )



    def _on_disconnect(
        self,
        client,
        userdata,
        rc
    ):


        self._connected.clear()


        print(
            "MQTT disconnected"
        )



    # ======================================================
    # RECEIVE MESSAGE FROM ESP32
    # ======================================================


    def _on_message(
        self,
        client,
        userdata,
        message
    ):


        topic = message.topic


        payload = (
            message.payload
            .decode()
        )


        print(
            "MQTT MESSAGE"
        )

        print(
            topic,
            payload
        )



    # ======================================================
    # START MQTT
    # ======================================================


    def start(self):


        self.client.connect(
            config.MQTT_BROKER_HOST,
            config.MQTT_BROKER_PORT,
            60
        )


        self.client.loop_start()


        self._loop_started = True



    # ======================================================
    # SEND COMMAND TO ESP32
    # ======================================================


    def publish_command(
        self,
        classification: str,
        confidence: float
    ):


        data = {


            "class":
            classification,


            "confidence":
            confidence


        }



        payload = json.dumps(
            data
        )



        self.client.publish(

            "sampah/command",

            payload,

            qos=1

        )



        print(
            "Command sent:",
            payload
        )

    
   def publish_prediction(
     self,
     prediction_result: dict
   ):

     payload = json.dumps(
         prediction_result
     )

    self.client.publish(
        "sampah/command",
        payload,
        qos=1
    )

    print(
        "Prediction sent:",
        payload
    )

    # ======================================================
    # STOP
    # ======================================================


    def stop(self):


        if self._loop_started:

            self.client.loop_stop()


        self.client.disconnect()
