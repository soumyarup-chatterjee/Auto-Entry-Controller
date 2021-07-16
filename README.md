The script developed as a part of this project detects the temperature and the presence of a face mask.

Technologies used to achieve said objective include: Convolutional Neural Network using TensorFlow (MobileNetV2), and Arduino.

If the temperature of a person is of an allowable value, it moves on to checking if the person is wearing a mask.
Only if both the temperature and mask criteria is fulfilled, a PASS signal is generated. Otherwise the resultant signal is FAIL.

The generated signal is then passed to the Arduino which controls the state of a digital pin.
The state of the digital pin can be used as a trigger to control the necessary actuation mechanism in a real-world scenario.
