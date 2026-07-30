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


        self.result_callback = (
            result_callback
        )


        self._connected = (
            threading.Event()
        )


        self._loop_started = False


        self.client = (
            self._create_client()
        )



    # =====================================
    # CREATE CLIENT
    # =====================================

    def _create_client(self):


        client = mqtt.Client(
            client_id=
            config.MQTT_CLIENT_ID
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



    # =====================================
    # CONNECT CALLBACK
    # =====================================

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



    # =====================================
    # RECEIVE MESSAGE
    # =====================================

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
            "MQTT MESSAGE:",
            message.topic,
            payload
        )



    # =====================================
    # START
    # =====================================

    def start(
        self
    ):


        self.client.connect(
            config.MQTT_BROKER_HOST,
            config.MQTT_BROKER_PORT,
            config.MQTT_KEEPALIVE_SECONDS
        )


        self.client.loop_start()


        self._loop_started = True



    # =====================================
    # STATUS
    # =====================================

    def is_connected(
        self
    ):


        return (
            self._connected.is_set()
        )



    # =====================================
    # SEND RESULT TO ESP32
    # =====================================

    def publish_prediction_command(
        self,
        prediction_result: dict[str, Any]
    ):


        if not self.is_connected():

            print(
                "MQTT belum terhubung"
            )

            return False



        label = prediction_result.get(
            "label",
            "NO_OBJECT"
        )


        payload = json.dumps(
            {
                "class": label
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



        else:


            print(
                "Publish gagal"
            )


            return False



    # =====================================
    # STOP
    # =====================================

    def stop(
        self
    ):


        if self._loop_started:

            self.client.loop_stop()



        self.client.disconnect()
