
import cv2
import mediapipe as mp
import snap7
import math
import time


# ==========================================================
# PLC SETTINGS
# ==========================================================

PLC_IP = "192.168.0.1"

RACK = 0
SLOT = 1

DB_NUMBER = 1

# DB1.DBX0.1 = Motor_Command
MOTOR_BYTE = 0
MOTOR_BIT = 1


# ==========================================================
# CONNECT TO PLC
# ==========================================================

print("Connecting to PLC...")

plc = snap7.Client()

try:
    plc.connect(PLC_IP, RACK, SLOT)

    if not plc.get_connected():
        print("ERROR: Could not connect to PLC")
        exit()

    print("PLC connected successfully!")
    print("PLC IP:", PLC_IP)

except Exception as e:
    print("PLC connection error:")
    print(e)
    exit()


# ==========================================================
# MEDIAPIPE SETUP
# ==========================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ==========================================================
# CAMERA
# ==========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Camera could not be opened!")

    plc.disconnect()
    exit()


# ==========================================================
# MOTOR FUNCTION
# ==========================================================

last_motor_state = None


def motor_control(state):

    global last_motor_state

    # Don't write to PLC if state hasn't changed
    if state == last_motor_state:
        return

    try:

        # Read current byte
        data = plc.db_read(
            DB_NUMBER,
            MOTOR_BYTE,
            1
        )

        if state:

            # Set DBX0.1 = 1
            data[0] |= (1 << MOTOR_BIT)

            print("✋ OPEN HAND -> DB1.DBX0.1 = 1")


        else:

            # Set DBX0.1 = 0
            data[0] &= ~(1 << MOTOR_BIT)

            print("✊ CLOSED/NO HAND -> DB1.DBX0.1 = 0")


        # Write byte back to PLC
        plc.db_write(
            DB_NUMBER,
            MOTOR_BYTE,
            data
        )

        last_motor_state = state


    except Exception as e:

        print("PLC write error:")

        print(e)

        # Communication failed
        last_motor_state = None


# ==========================================================
# HAND OPEN/CLOSED DETECTION
# ==========================================================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def detect_open_hand(hand):

    wrist = hand.landmark[0]

    # Fingertips
    fingertips = [
        8,   # Index
        12,  # Middle
        16,  # Ring
        20   # Pinky
    ]

    extended = 0

    for tip in fingertips:

        # Distance fingertip -> wrist
        tip_distance = distance(
            hand.landmark[tip],
            wrist
        )

        # Distance PIP joint -> wrist
        pip_distance = distance(
            hand.landmark[tip - 2],
            wrist
        )

        if tip_distance > pip_distance:

            extended += 1


    # If at least 3 fingers are extended,
    # consider the hand OPEN.

    if extended >= 3:

        return True

    return False


# ==========================================================
# MAIN LOOP
# ==========================================================

print("")
print("==============================")
print(" HAND MOTOR CONTROL")
print("==============================")
print("OPEN HAND   -> DB1.DBX0.1 = 1")
print("CLOSED HAND -> DB1.DBX0.1 = 0")
print("NO HAND     -> DB1.DBX0.1 = 0")
print("ESC         -> EXIT")
print("==============================")
print("")


try:

    while True:

        # --------------------------------------------------
        # Get camera frame
        # --------------------------------------------------

        ret, frame = camera.read()

        if not ret:

            print("Camera error!")

            motor_control(False)

            continue


        # Mirror image
        frame = cv2.flip(frame, 1)


        # --------------------------------------------------
        # Convert image
        # --------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # --------------------------------------------------
        # Detect hand
        # --------------------------------------------------

        results = hands.process(rgb)


        # --------------------------------------------------
        # HAND FOUND
        # --------------------------------------------------

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]


            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )


            # Detect open/closed
            open_hand = detect_open_hand(hand)


            # ----------------------------------------------
            # OPEN HAND
            # ----------------------------------------------

            if open_hand:

                motor_control(True)

                text = "OPEN HAND - DB1.DBX0.1 = 1"


            # ----------------------------------------------
            # CLOSED HAND
            # ----------------------------------------------

            else:

                motor_control(False)

                text = "CLOSED HAND - DB1.DBX0.1 = 0"


        # --------------------------------------------------
        # NO HAND
        # --------------------------------------------------

        else:

            motor_control(False)

            text = "NO HAND - DB1.DBX0.1 = 0"


        # --------------------------------------------------
        # Display status
        # --------------------------------------------------

        cv2.putText(
            frame,
            text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # Show camera
        cv2.imshow(
            "Hand Motor Control",
            frame
        )


        # ESC = exit
        key = cv2.waitKey(1) & 0xFF

        if key == 27:

            break


# ==========================================================
# STOP EVERYTHING
# ==========================================================

finally:

    print("")
    print("Stopping motor...")

    try:

        motor_control(False)

    except:

        pass


    camera.release()

    cv2.destroyAllWindows()


    if plc.get_connected():

        plc.disconnect()


    print("PLC disconnected.")
    print("Program finished.")


