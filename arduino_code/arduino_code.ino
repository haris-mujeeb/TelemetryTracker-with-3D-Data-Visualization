#include <Wire.h>                             //Needed for I2C to GNSS
#include "SparkFun_BNO08x_Arduino_Library.h"  // Click here to get the library: http://librarymanager/All#SparkFun_BNO080
#include <Preferences.h>
#include <ezButton.h>
#include <SPI.h>
#include <SD.h>

//#define DEBUG_ON

struct sensor_data {
  float accel_X;
  float accel_Y;
  float accel_Z;
  float gyro_X;
  float gyro_Y;
  float gyro_Z;
  float mag_X;
  float mag_Y;
  float mag_Z;
  float temp;
  long latitude;
  long longitude;
  long altitude;
  int siv;
  int force;
  long gps_xang_rate;
  long gps_yang_rate;
  long gps_zang_rate;
  long gps_xaccel;
  long gps_yaccel;
  long gps_zaccel;

  long NedNorthVel, NedEastVel, NedDownVel, VerticalAccEst, HorizontalAccEst, SpeedAccEst, HeadingAccEst;
  int headVeh, magDec, magAcc;


  unsigned long time_gps;
  unsigned long time_imu;
  unsigned long time_pressure;
};

File myFile;
sensor_data sensor;
Preferences preferences;

#include <SparkFun_u-blox_GNSS_Arduino_Library.h>  //Click here to get the library: http://librarymanager/All#SparkFun_u-blox_GNSS
SFE_UBLOX_GNSS myGNSS;
unsigned int file_counter = 0;
bool measuring_state = false;

#define PRESSURE_SENSOR_PIN A0
#define BNO08X_INT 9
#define BNO08X_RST 12


#define BNO08X_ADDR 0x4A
#define SCL_EXTRA 6
#define SDA_EXTRA 5
#define CALBUTTON A1
#define MEASBUTTON A2
#define LED_RED 10
#define LED_YELLOW 11

#define SCK_CARD A5

ezButton calbutton(CALBUTTON);    // create ezButton object that attach to pin 7;
ezButton measbutton(MEASBUTTON);  // create ezButton object that attach to pin 7;
BNO08x myIMU;

void save_data_tofile() {
  // if the file opened okay, write to it:
  if (myFile) {
    //String message = String(millis()) + "," + String(sensor.time_imu) + "," + String(sensor.accel_X) + "," + String(sensor.accel_Y) + "," + String(sensor.accel_Z) + "," + String(sensor.gyro_X) + "," + String(sensor.gyro_Y) + "," + String(sensor.gyro_Z) + "," + String(sensor.mag_X) + "," + String(sensor.mag_Y) + "," + String(sensor.mag_Z) + "," + String(sensor.temp) + "," + String(sensor.time_gps) + "," + String(sensor.latitude) + "," + String(sensor.longitude) + "," + String(sensor.altitude) + "," + String(sensor.siv) + "," + String(sensor.time_pressure) + "," + String(sensor.force);
    myFile.flush();  // this should be before writing so we can eliminate the possibility of waiting for any transfers to complete before the write happens
    myFile.printf(
      "%lu,%lu,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.2f,%lu,%ld,%ld,%ld,%d,"
      "%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%d,%d\n",

      millis(),
      sensor.time_imu,
      sensor.accel_X, sensor.accel_Y, sensor.accel_Z,
      sensor.gyro_X, sensor.gyro_Y, sensor.gyro_Z,
      sensor.mag_X, sensor.mag_Y, sensor.mag_Z,
      sensor.temp,
      sensor.time_gps,
      sensor.latitude,
      sensor.longitude,
      sensor.altitude,
      sensor.siv,
      sensor.NedNorthVel,
      sensor.NedEastVel,
      sensor.NedDownVel,
      sensor.VerticalAccEst,
      sensor.HorizontalAccEst,
      sensor.SpeedAccEst,
      sensor.HeadingAccEst,
      sensor.headVeh,
      sensor.magDec,
      sensor.magAcc,
      sensor.time_pressure,
      sensor.force);

    //myFile.println(message);
  }
  // if the file didn't open, print an error:
  else {
    Serial.println("error opening file ");
  }
}

void getdata_GPS(void* pvParameters) {
  unsigned long lastTime_GPS = millis();
  long latitude, longitude, altitude, NedNorthVel, NedEastVel, NedDownVel, VerticalAccEst, HorizontalAccEst, SpeedAccEst, HeadingAccEst;
  int headVeh, magDec, magAcc;
  byte SIV;
  SFE_UBLOX_GNSS* gnss = static_cast<SFE_UBLOX_GNSS*>(pvParameters);
  while (1) {

    if (measuring_state && gnss->getPVT() && (gnss->getInvalidLlh() == false)) {  //200 msec for GPS update
      lastTime_GPS = millis();                                                    //Update the timer
      latitude = gnss->getLatitude();
      longitude = gnss->getLongitude();
      altitude = gnss->getAltitude();
      SIV = gnss->getSIV();
      NedNorthVel = gnss->getNedNorthVel();
      NedEastVel = gnss->getNedEastVel();
      NedDownVel = gnss->getNedDownVel();
      VerticalAccEst = gnss->getVerticalAccEst();
      HorizontalAccEst = gnss->getHorizontalAccEst();
      SpeedAccEst = gnss->getSpeedAccEst();
      HeadingAccEst = gnss->getHeadingAccEst();

      if (gnss->getHeadVehValid() == true) {
        headVeh = gnss->getHeadVeh();
        magDec = gnss->getMagDec();
        magAcc = gnss->getMagAcc();
      }

      sensor.time_gps = millis();

      sensor.latitude = latitude;
      sensor.longitude = longitude;
      sensor.altitude = altitude;
      sensor.siv = SIV;

      sensor.NedNorthVel = NedNorthVel;
      sensor.NedEastVel = NedEastVel;
      sensor.NedDownVel = NedDownVel;
      sensor.VerticalAccEst = VerticalAccEst;
      sensor.HorizontalAccEst = HorizontalAccEst;
      sensor.SpeedAccEst = SpeedAccEst;
      sensor.HeadingAccEst = HeadingAccEst;
      sensor.headVeh = headVeh;
      sensor.magDec = magDec;
      sensor.magAcc = magAcc;


      gnss->flushPVT();
    }
    delay(10);
  }
}

void setReports(void) {
  Serial.println("Setting desired reports");
  if (myIMU.enableMagnetometer() == true) {
    Serial.println(F("Magnetometer enabled"));
  } else {
    Serial.println("Could not enable magnetometer");
  }

  if (myIMU.enableAccelerometer() == true) {
    Serial.println(F("Accelerometer enabled"));
  } else {
    Serial.println("Could not enable Accelerometer");
  }

  if (myIMU.enableGyro() == true) {
    Serial.println(F("Gyro enabled"));
  } else {
    Serial.println("Could not enable Gyro");
  }
}

void setup() {
  Serial.begin(115200);
  preferences.begin("trax", false);
  pinMode(LED_RED, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);

  digitalWrite(LED_RED, HIGH);
  digitalWrite(LED_YELLOW, HIGH);


  delay(1000);  // Wait for Serial to become available.
  Serial.println("Starting surfsense");
  pinMode(PRESSURE_SENSOR_PIN, INPUT);

  Wire.begin();
  Wire1.begin(SDA_EXTRA, SCL_EXTRA);

  file_counter = preferences.getUInt("counter", 0);

  if (!SD.begin(SCK_CARD)) {
    Serial.println("initialization failed!");
    return;
  }


  if (myIMU.begin(BNO08X_ADDR, Wire1, BNO08X_INT, BNO08X_RST) == false) {
    Serial.println("BNO08x not detected at default I2C address. Check your jumpers and the hookup guide. Freezing...");
    while (1)
      ;
  }
  Serial.println("BNO08x found!");

  setReports();

  if (myGNSS.begin(Wire) == false) {
    Serial.println(F("u-blox GNSS not detected at default I2C address. Please check wiring. Freezing."));
    while (1)
      ;
  }

  myGNSS.setI2COutput(COM_TYPE_UBX);  //Set the I2C port to output both NMEA and UBX messages
  //myGNSS.saveConfigSelective(VAL_CFG_SUBSEC_IOPORT);  //Save (only) the communications port settings to flash and BBR

  //measurement time (inms)
  if (myGNSS.setMeasurementRate(60) == false) {
    Serial.println(F("Could not set the measurement rate. Freezing."));
    while (1)
      ;
  }

  myGNSS.setI2CpollingWait(25);

  //number of measurements to average to make a solution (higher is better quality position but takes more time for measurements to load)
  if (myGNSS.setNavigationRate(5) == false) {
    Serial.println(F("Could not set the navigation rate. Freezing."));
    while (1)
      ;
  }

  myGNSS.setAutoPVT(true);  //Tell the GNSS to "send" each solution

  byte rate = myGNSS.getNavigationFrequency();  //Get the update rate of this module
  Serial.print("Current update rate: ");
  Serial.println(rate);

  xTaskCreatePinnedToCore(
    getdata_GPS,    // Function
    "getdata_GPS",  // Name
    5000,           // Stack size
    &myGNSS,        // 👈 Pointer to pass
    1,              // Priority
    NULL,           // Task handle
    0               // Core
  );

  delay(500);
  digitalWrite(LED_RED, LOW);
  digitalWrite(LED_YELLOW, LOW);
}

void getdata_IMU() {
  if (myIMU.getSensorEvent() == true) {
    sensor.time_imu = millis();

    if (myIMU.getSensorEventID() == SENSOR_REPORTID_GYROSCOPE_CALIBRATED) {
      sensor.gyro_X = myIMU.getGyroX();
      sensor.gyro_Y = myIMU.getGyroY();
      sensor.gyro_Z = myIMU.getGyroZ();
    }

    if (myIMU.getSensorEventID() == SENSOR_REPORTID_ACCELEROMETER) {
      sensor.accel_X = myIMU.getAccelX();
      sensor.accel_Y = myIMU.getAccelY();
      sensor.accel_Z = myIMU.getAccelZ();
    }

    if (myIMU.getSensorEventID() == SENSOR_REPORTID_MAGNETIC_FIELD) {
      sensor.mag_X = myIMU.getMagX();
      sensor.mag_Y = myIMU.getMagY();
      sensor.mag_Z = myIMU.getMagZ();
    }
  }
}


void displaysensor_data() {
  Serial.print(">>sns dat:");
  Serial.print(millis());
  Serial.print(",IMU-ticks:");
  Serial.print(sensor.time_imu);
  Serial.print(",aX:");
  Serial.print(sensor.accel_X);
  Serial.print(",aY:");
  Serial.print(sensor.accel_Y);
  Serial.print(",aZ:");
  Serial.print(sensor.accel_Z);
  Serial.print(",gX:");
  Serial.print(sensor.gyro_X);
  Serial.print(",gY:");
  Serial.print(sensor.gyro_Y);
  Serial.print(",gZ:");
  Serial.print(sensor.gyro_Z);
  Serial.print(",mX:");
  Serial.print(sensor.mag_X);
  Serial.print(",mY:");
  Serial.print(sensor.mag_Y);
  Serial.print(",mZ:");
  Serial.print(sensor.mag_Z);
  Serial.print(",tX:");
  Serial.print(sensor.temp);
  Serial.print(",GPS-ticks:");
  Serial.print(sensor.time_gps);
  Serial.print(",lat:");
  Serial.print(sensor.latitude);
  Serial.print(",long:");
  Serial.print(sensor.longitude);
  Serial.print(",alt:");
  Serial.print(sensor.altitude);
  Serial.print(",siv:");
  Serial.print(sensor.siv);
  Serial.print(",pressure-ticks:");
  Serial.print(sensor.time_pressure);
  Serial.print(",force:");
  Serial.print(sensor.force);
  Serial.print("<<");
  Serial.println();
}


unsigned long lastTime_IMU;

void loop() {
  calbutton.loop();
  measbutton.loop();

  if (calbutton.isPressed()) {
    digitalWrite(LED_YELLOW, HIGH);
    Serial.println("Calbutton is pressed");
    if (myIMU.setCalibrationConfig(SH2_CAL_ACCEL || SH2_CAL_GYRO || SH2_CAL_MAG) == true) {
      Serial.println(F("Calibration Command Sent Successfully"));
    } else {
      Serial.println(F("Could not send Calibration Command. Freezing..."));
      while (1) delay(10);
    }
    //wait 10 seconds and save calibrated data to bno085 flash
    delay(10000);
    if (myIMU.saveCalibration() == true) {
      Serial.println(F("Calibration data was saved successfully"));
    } else {
      Serial.println("Save Calibration Failure");
    }

    digitalWrite(LED_YELLOW, LOW);
  }

  if (measbutton.isPressed()) {
    Serial.println("Measbutton is pressed");
    if (myFile) myFile.flush();
    if (myFile) myFile.close();
    if (!measuring_state) {
      digitalWrite(LED_RED, HIGH);
      preferences.putUInt("counter", ++file_counter);
      String filename = "/trax_" + String(myGNSS.getYear()) + "_" + String(myGNSS.getMonth()) + "_" + String(myGNSS.getDay()) + "_" + String(myGNSS.getHour()) + "_" + String(myGNSS.getMinute()) + "_" + String(myGNSS.getSecond()) + "_#" + String(file_counter) + ".csv";
      myFile = SD.open(filename, FILE_WRITE);
      if (myFile) Serial.println("File opened for Writing");
      myFile.println(F(
        "timestamp,IMU-ticks,aX,aY,aZ,gX,gY,gZ,mX,mY,mZ,tX,"
        "GPS-ticks,lat,long,alt,siv,"
        "NedNorthVel,NedEastVel,NedDownVel,"
        "VerticalAccEst,HorizontalAccEst,SpeedAccEst,HeadingAccEst,"
        "headVeh,magDec,magAcc,"
        "pressure-ticks,force1"));
      measuring_state = true;
    } else {
      digitalWrite(LED_RED, LOW);
      measuring_state = false;
    }
  }

  if (measuring_state) {
    if (myIMU.wasReset()) {
      Serial.print("Sensor was reset");
      setReports();
    }

    if (millis() - lastTime_IMU > 10) {  //10 msec for IMU update
      lastTime_IMU = millis();           //Update the timer
      getdata_IMU();
      sensor.force = analogRead(PRESSURE_SENSOR_PIN);
      sensor.time_pressure = millis();
      save_data_tofile();
#ifdef DEBUG_ON
      displaysensor_data();
#endif
    }
  }

  delay(10);  //Don't pound too hard on the I2C bus
}
