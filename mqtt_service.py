from __future__ import annotations

import json
import ssl
import threading
from typing import Any, Callable

import paho.mqtt.client as mqtt

import config


MessageCallback = Callable[
    [str, bytes],
    None,
]


class MQTTService:
    """
    MQTT service Raspberry Pi.

    Fungsi:
    - menerima hasil klasifikasi AI
    - menyimpan hasil terakhir
    - mengirim command ke ESP32

    Alur:
        Raspberry Pi
        -> MQTT Broker
        -> ESP32
    """


    def __init__(
        self,
        subscribe_topic: str = "sampah/#",
        message_callback: MessageCallback | None = None,
    ) -> None:


        self.subscribe_topic = subscribe_topic

        self.message_callback = message_callback


        self._connected = threading.Event()


        self._latest_message: dict[str, Any] | None = None


        self._latest_result: dict[str, Any] | None = None


        self._loop_started = False


        self.client = self._create_client()



    # ========================================================
    # CREATE MQTT CLIENT
    # ========================================================

    def _create_client(
        self,
    ) -> mqtt.Client:


        client_id = (
            f"raspberry-ai-{config.DEVICE_ID}"
        )


        try:

            client = mqtt.Client(
                callback_api_version=(
                    mqtt.CallbackAPIVersion.VERSION2
                ),
                client_id=client_id,
                protocol=mqtt.MQTTv311,
                transport="websockets",
            )


        except (
            AttributeError,
            TypeError,
        ):

            client = mqtt.Client(
                client_id=client_id,
                protocol=mqtt.MQTTv311,
                transport="websockets",
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



        client.ws_set_options(
            path=config.MQTT_WEBSOCKET_PATH
        )



        client.tls_set(
            ca_certs=config.MQTT_CA_CERT,
            tls_version=ssl.PROTOCOL_TLS_CLIENT,
        )



        if config.MQTT_USERNAME:


            client.username_pw_set(
                username=config.MQTT_USERNAME,
                password=config.MQTT_PASSWORD,
            )



        client.reconnect_delay_set(
            min_delay=config.MQTT_RECONNECT_MIN_SECONDS,
            max_delay=config.MQTT_RECONNECT_MAX_SECONDS,
        )


        return client



    # ========================================================
    # CONNECT CALLBACK
    # ========================================================

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any,
        reason_code: Any,
        properties: Any = None,
    ) -> None:


        code = self._reason_code_to_int(
            reason_code
        )



        if code != 0:

            self._connected.clear()

            print(
                "MQTT gagal terhubung:",
                reason_code
            )

            return



        self._connected.set()



        print(
            "MQTT berhasil terhubung"
        )


        print(
            f"Broker: "
            f"{config.MQTT_BROKER_HOST}:"
            f"{config.MQTT_BROKER_PORT}"
        )



        result, message_id = (
            client.subscribe(
                topic=self.subscribe_topic,
                qos=1,
            )
        )



        if result == mqtt.MQTT_ERR_SUCCESS:

            print(
                "Subscribe berhasil:",
                self.subscribe_topic
            )

        else:

            print(
                "Subscribe gagal:",
                result
            )



    # ========================================================
    # DISCONNECT CALLBACK
    # ========================================================

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        *args: Any,
    ) -> None:


        self._connected.clear()


        print(
            "MQTT disconnected"
        )



    # ========================================================
    # RECEIVE MESSAGE
    # ========================================================

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: Any,
        message: mqtt.MQTTMessage,
    ) -> None:


        print()

        print(
            "=" * 50
        )


        print(
            "Topic:",
            message.topic
        )



        try:


            payload_text = (
                message.payload
                .decode("utf-8")
            )



            data = json.loads(
                payload_text
            )



            self._latest_message = data



            # =====================================
            # SIMPAN HASIL KLASIFIKASI AI
            # =====================================

            if (

                "label" in data

                and

                "confidence" in data

            ):


                self._latest_result = data



                print(
                    "Classification result:"
                )


                print(
                    json.dumps(
                        data,
                        indent=2
                    )
                )



        except Exception as error:


            print(
                "Payload bukan JSON:",
                error
            )


            print(
                message.payload
            )



        if self.message_callback is not None:


            try:


                self.message_callback(
                    message.topic,
                    message.payload,
                )


            except Exception as error:


                print(
                    "Callback error:",
                    error
                )



    # ========================================================
    # START SERVICE
    # ========================================================

    def start(
        self,
        connection_timeout_seconds: float = 15.0,
    ) -> bool:


        if self._loop_started:

            return self.is_connected()



        print(
            "Starting MQTT service..."
        )


        self.client.connect_async(

            host=config.MQTT_BROKER_HOST,

            port=config.MQTT_BROKER_PORT,

            keepalive=config.MQTT_KEEPALIVE_SECONDS,

        )



        self.client.loop_start()



        self._loop_started = True



        connected = (
            self._connected.wait(
                timeout=connection_timeout_seconds
            )
        )


        return connected



    # ========================================================
    # PUBLISH COMMAND KE ESP32
    # ========================================================

    def publish_prediction_command(
        self,
        result: dict[str, Any],
    ) -> bool:



        if not self.is_connected():

            print(
                "MQTT belum terhubung"
            )

            return False



        payload = json.dumps(

            {

                "class": result.get(
                    "label",
                    "Unknown"
                ),


                "confidence": result.get(
                    "confidence",
                    0.0
                ),


                "frame_id": result.get(
                    "frame_id",
                    "-"
                ),

            }

        )



        info = self.client.publish(

            topic="sampah/command",

            payload=payload,

            qos=1,

        )



        if (
            info.rc
            ==
            mqtt.MQTT_ERR_SUCCESS
        ):


            print(
                "Command ESP32 terkirim:"
            )


            print(
                payload
            )


            return True



        print(
            "Gagal mengirim command"
        )


        return False



    # ========================================================
    # STATUS
    # ========================================================

    def is_connected(
        self,
    ) -> bool:


        return self._connected.is_set()



    def get_latest_message(
        self,
    ) -> dict[str, Any] | None:


        if self._latest_message is None:

            return None


        return self._latest_message.copy()



    def get_latest_result(
        self,
    ) -> dict[str, Any] | None:


        if self._latest_result is None:

            return None


        return self._latest_result.copy()



    # ========================================================
    # STOP
    # ========================================================

    def stop(
        self,
    ) -> None:


        if self.is_connected():

            self.client.disconnect()



        if self._loop_started:

            self.client.loop_stop()

            self._loop_started = False



        self._connected.clear()



        print(
            "MQTT service stopped"
        )



    # ========================================================
    # HELPER
    # ========================================================

    @staticmethod
    def _reason_code_to_int(
        reason_code: Any,
    ) -> int:


        try:

            return int(reason_code)


        except (
            TypeError,
            ValueError,
        ):


            return int(

                getattr(
                    reason_code,
                    "value",
                    -1
                )

            )




