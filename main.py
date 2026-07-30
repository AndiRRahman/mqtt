#include <Arduino.h>
#include <ArduinoJson.h>

#include "UltrasonicSensor.h"
#include "StepperMotor.h"

#include "wifi_manager.h"
#include "mqtt_client.h"


// ============================
// WIFI
// ============================

const char* WIFI_SSID = "UTeM-IOT";

const char* WIFI_PASSWORD = "!@utemIOT";


WiFiManager wifi(
    WIFI_SSID,
    WIFI_PASSWORD
);



// ============================
// MQTT
// ============================

const char* MQTT_SERVER = "10.132.6.38";

const int MQTT_PORT = 1883;


MQTTClient mqtt(
    MQTT_SERVER,
    MQTT_PORT,
    "ESP32-SAMPAH"
);



// ============================
// STEPPER
// ============================

const int STEPS_PER_REV = 2048;


StepperMotor myStepper(
    STEPS_PER_REV,
    19,
    22,
    21,
    23
);



// ============================
// SENSOR
// ============================

#define MAX_DISTANCE 200


UltrasonicSensor sensors[] =
{

    UltrasonicSensor(
        1,
        13,
        12,
        MAX_DISTANCE
    ),

    UltrasonicSensor(
        2,
        14,
        27,
        MAX_DISTANCE
    ),

    UltrasonicSensor(
        3,
        26,
        25,
        MAX_DISTANCE
    ),

    UltrasonicSensor(
        4,
        33,
        32,
        MAX_DISTANCE
    ),

    UltrasonicSensor(
        5,
        16,
        17,
        MAX_DISTANCE
    )

};



// ============================
// POSISI STEPPER
// ============================

const int POS_HOME = 0;

const int POS_PLASTIC = 1;

const int POS_PAPER = 2;

const int POS_CARDBOARD = 3;

const int POS_METAL = 4;

const int POS_FOOD = 5;



// =================================================
// MQTT CALLBACK
// =================================================

void receiveCommand(
    String topic,
    String message
)

{

    Serial.println();

    Serial.println(
        "=========================="
    );


    Serial.println(
        "MQTT COMMAND DITERIMA"
    );


    Serial.print(
        "Topic : "
    );

    Serial.println(
        topic
    );


    Serial.print(
        "Payload : "
    );

    Serial.println(
        message
    );



    StaticJsonDocument<256> doc;



    DeserializationError error =
        deserializeJson(
            doc,
            message
        );



    if(error)
    {

        Serial.println(
            "JSON ERROR"
        );

        return;

    }



    String kelas =
        doc["class"];



    float confidence =
        doc["confidence"];



    int class_id =
        doc["class_id"];





    Serial.print(
        "Class : "
    );

    Serial.println(
        kelas
    );



    Serial.print(
        "Confidence : "
    );

    Serial.println(
        confidence
    );



    Serial.print(
        "Class ID : "
    );

    Serial.println(
        class_id
    );



    // =====================================
    // CLASSIFICATION
    // KE POSISI STEPPER
    // =====================================


    if(kelas == "Plastic")
    {

        Serial.println(
            "AKSI : PLASTIC"
        );


        myStepper.moveToPosition(
            POS_PLASTIC
        );

    }



    else if(kelas == "Paper")
    {

        Serial.println(
            "AKSI : PAPER"
        );


        myStepper.moveToPosition(
            POS_PAPER
        );

    }



    else if(kelas == "Cardboard")
    {

        Serial.println(
            "AKSI : CARDBOARD"
        );


        myStepper.moveToPosition(
            POS_CARDBOARD
        );

    }



    else if(kelas == "Metal")
    {

        Serial.println(
            "AKSI : METAL"
        );


        myStepper.moveToPosition(
            POS_METAL
        );

    }



    else if(kelas == "Food")
    {

        Serial.println(
            "AKSI : FOOD"
        );


        myStepper.moveToPosition(
            POS_FOOD
        );

    }



    else if(kelas == "NO_OBJECT")
    {

        Serial.println(
            "AKSI : TIDAK ADA SAMPAH"
        );


        myStepper.moveToPosition(
            POS_HOME
        );

    }



    else
    {

        Serial.println(
            "CLASS TIDAK DIKENALI"
        );

    }



    Serial.println(
        "=========================="
    );

}



// ============================
// SETUP
// ============================

void setup()

{

    Serial.begin(
        115200
    );


    delay(1000);



    Serial.println(
        "START ESP32 SAMPAH"
    );



    wifi.begin();



    mqtt.begin();



    mqtt.setCallback(
        receiveCommand
    );



    mqtt.subscribe(
        "sampah/command"
    );



    myStepper.init(
        15
    );



    Serial.println(
        "ESP32 READY"
    );

}



// ============================
// LOOP
// ============================

void loop()

{

    mqtt.loop();

}
