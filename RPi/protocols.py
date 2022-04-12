## Communication protocols, to allow all subsystems to know the common ways of communication.

MESSAGE_SEPARATOR = '|'.encode()
NEWLINE = '\n'.encode()

ANDROID_HEADER = 'AND'.encode()
RPI_HEADER = 'RPI'.encode()
ALGORITHM_HEADER = 'ALG'.encode()
STM_HEADER = 'STM'.encode()

class Status:
    IDLE = 'idle'.encode()
    EXPLORING = 'exploring'.encode()
    SHORTEST_PATH = 'shortest path'.encode()


class Android_RPI: ## should it be android to rpi then to stm or android to stm straight
    MOVE_FORWARD = 'U1|'.encode()
    MOVE_BACK = 'B1|'.encode()
    TURN_LEFT = 'L1|'.encode()
    TURN_RIGHT = 'W1|'.encode()
    DO_SHORTCUT_1 = 'F1|'.encode()
    DO_SHORTCUT_2 = 'F2|'.encode()

    ALL_MESSAGES = [
        MOVE_FORWARD,
        MOVE_BACK,
        TURN_LEFT,
        TURN_RIGHT,
        DO_SHORTCUT_1,
        DO_SHORTCUT_2,
    ]


class Android_Algorithm:
    START_EXPLORATION = 'SE|'.encode()
    START_SHORTEST_PATH = 'SP|'.encode()
    SEND_ARENA = 'SendArena'.encode()
    


class Android_RPI: 
    CALIBRATE_SENSOR = 'CS|'.encode()


class Algorithm_Android:
    MOVE_FORWARD = 'FS'.encode()[0]
    TURN_LEFT = 'L'.encode()[0]
    TURN_RIGHT = 'W'.encode()[0]
    CALIBRATING_CORNER = 'C'.encode()[0]
    SENSE_ALL = 'S'.encode()[0]
    ALIGN_RIGHT = 'AR'.encode()[0]
    ALIGN_FRONT = 'AF'.encode()[0]

    MDF_STRING = 'M'.encode()[0]


class Algorithm_RPi:
    TAKE_PICTURE = 'C'.encode()[0]
    EXPLORATION_COMPLETE = 'D'.encode()


class RPI_Android:
    STATUS_EXPLORATION = '{"status":"exploring"}'.encode()
    STATUS_SHORTEST_PATH = '{"status":"shortest path"}'.encode()
    STATUS_TURN_LEFT = '{"status":"turning left"}'.encode()
    STATUS_TURN_RIGHT = '{"status":"turning right"}'.encode()
    STATUS_IDLE = '{"status":"idle"}'.encode()
    STATUS_TAKING_PICTURE = '{"status":"taking picture"}'.encode()
    STATUS_CALIBRATING_CORNER = '{"status":"calibrating corner"}'.encode()
    STATUS_SENSE_ALL = '{"status":"sense all"}'.encode()
    STATUS_MOVING_FORWARD = '{"status":"moving forward"}'.encode()
    STATUS_ALIGN_RIGHT = '{"status":"align right"}'.encode()
    STATUS_ALIGN_FRONT = '{"status":"align front"}'.encode()

    MOVE_UP = '{"move":[{"direction":"forward"}]}'.encode()
    TURN_LEFT = '{"move":[{"direction":"left"}]}'.encode()
    TURN_RIGHT = '{"move":[{"direction":"right"}]}'.encode()


class RPI_STM:
    CALIBRATE_SENSOR = 'L|A|'.encode()
    START_EXPLORATION = 'E|'.encode()
    START_SHORTEST_PATH = 'F|'.encode()


class RPI_Algorithm:
    DONE_TAKING_PICTURE = 'D'.encode()
    DONE_IMG_REC = 'I'.encode()