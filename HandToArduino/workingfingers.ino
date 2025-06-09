#include <Servo.h>

Servo servos[5];
int pins[5] = {3, 4, 5, 6, 7};

// Servo types: true for degree-based (D3, D4), false for state-based
bool servoTypes[5] = {false, true, true, false, false};  // Index 1 and 2 are degree-based

// For state-based servos - timed movement logic
unsigned long stopTimes[5] = {0};
bool isMoving[5] = {false};
int lastStateValues[5] = {0, 90, 90, 0, 0};  // Track last values (0=open, 180=closed for state-based)

// Movement parameters for state-based servos
int moveDuration = 500;  // ms to move between states

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < 5; i++) {
    servos[i].attach(pins[i]);
    servos[i].write(90);  // Start stopped/neutral for all servos
  }
  Serial.println("Arduino Ready - Timed Movement (Open/Closed Only)");
}

void loop() {
  unsigned long currentTime = millis();
  
  // Check if any state-based servo needs to stop
  for (int i = 0; i < 5; i++) {
    if (!servoTypes[i] && isMoving[i] && currentTime >= stopTimes[i]) {
      servos[i].write(90);  // Stop movement
      isMoving[i] = false;
      stopTimes[i] = 0;
    }
  }

  // Process new commands
  if (Serial.available()) {
    String data = Serial.readStringUntil('\n');
    data.trim();
    
    // Parse command: finger,type,value
    int commaIndex1 = data.indexOf(',');
    int commaIndex2 = data.lastIndexOf(',');
    
    if (commaIndex1 > 0 && commaIndex2 > commaIndex1) {
      int finger = data.substring(0, commaIndex1).toInt();
      String commandType = data.substring(commaIndex1 + 1, commaIndex2);
      int value = data.substring(commaIndex2 + 1).toInt();
      
      if (finger >= 0 && finger < 5) {
        if (servoTypes[finger]) {
          // Degree-based servo (D3, D4) - direct angle control
          if (commandType == "angle") {
            int constrainedAngle = constrain(value, 0, 180);
            servos[finger].write(constrainedAngle);
            Serial.print("Finger ");
            Serial.print(finger);
            Serial.print(" (degree-based) set to ");
            Serial.print(constrainedAngle);
            Serial.println("°");
          }
        } else {
          // State-based servo - timed movement only on state change
          if (commandType == "state" && value != lastStateValues[finger]) {
            // Stop any current movement first
            servos[finger].write(90);
            isMoving[finger] = false;
            
            int moveAngle;
            if (value == 0) {
              // Open state
              moveAngle = 60;   // Counter-clockwise
              Serial.print("Finger ");
              Serial.print(finger);
              Serial.println(" (state-based) -> OPEN");
            } else if (value == 180) {
              // Closed state
              moveAngle = 120;  // Clockwise
              Serial.print("Finger ");
              Serial.print(finger);
              Serial.println(" (state-based) -> CLOSED");
            } else {
              moveAngle = 90;   // No movement
            }
            
            if (moveAngle != 90) {
              servos[finger].write(moveAngle);
              stopTimes[finger] = currentTime + moveDuration;
              isMoving[finger] = true;
            }
            
            lastStateValues[finger] = value;
          }
        }
      }
    }
  }
}

// Optional: Function to get servo status
void printServoStatus() {
  Serial.println("--- Servo Status ---");
  for (int i = 0; i < 5; i++) {
    Serial.print("Finger ");
    Serial.print(i);
    Serial.print(": ");
    if (servoTypes[i]) {
      Serial.println("Degree-based (with stopper)");
    } else {
      Serial.print("State-based, Last state: ");
      Serial.print(lastStateValues[i] == 0 ? "OPEN" : "CLOSED");
      if (isMoving[i]) {
        Serial.print(" (Currently moving, stops at ");
        Serial.print(stopTimes[i]);
        Serial.print(")");
      } else {
        Serial.print(" (Stopped)");
      }
      Serial.println();
    }
  }
  Serial.println("-------------------");
}