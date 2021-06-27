
#include <Wire.h>
#include <Adafruit_MLX90614.h>

Adafruit_MLX90614 mlx = Adafruit_MLX90614();

void setup() {
  pinMode(12,OUTPUT);
  Serial.begin(9600); 

  mlx.begin();  
}

void loop() {
  Serial.print(mlx.readAmbientTempC()); 
  Serial.print("x"); Serial.print(mlx.readObjectTempC());

  Serial.println();
  
  if(Serial.available() > 0){
    if(Serial.readString().toInt() == 1){
      digitalWrite(12,HIGH);
    }
  }  
  else{
    digitalWrite(12,LOW);
  }
  delay(2000);
}
