from __future__ import annotations

import json
import threading
from typing import Any

import paho.mqtt.client as mqtt

import config


class MQTTService:


    def __init__(self):

        self._connected = threading.Event()

        self._loop_started = False

        self.client = self._create_client()



    # ======================================================
    # CREATE CLIENT
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
    # RECEIVE MESSAGE
    # ======================================================

    def _on_message(
        self,
        client,
        userdata,
        message
    ):

        payload = (
            message.payload
            .decode()
        )


        print(
            "MQTT MESSAGE"
        )

        print(
            message.topic,
            payload
        )



    # ======================================================
    # START MQTT
    # ======================================================

    def start(
        self,
        timeout=10
    ):


        self.client.connect(
            config.MQTT_BROKER_HOST,
            config.MQTT_BROKER_PORT,
            60
        )


        self.client.loop_start()


        self._loop_started = True



        connected = (
            self._connected.wait(
                timeout
            )
        )


        if not connected:

            print(
                "MQTT connection timeout"
            )



        return connected



    # ======================================================
    # SEND AI RESULT TO ESP32
    # ======================================================

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
                    "NO_OBJECT"
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
    # STATUS
    # ======================================================

    def is_connected(
        self
    ):

        return self._connected.is_set()



    # ======================================================
    # STOP
    # ======================================================

    def stop(
        self
    ):


        if self._loop_started:

            self.client.loop_stop()


        self.client.disconnect()


        self._connected.clear()
