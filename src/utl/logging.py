from datetime import datetime


class Log:
    def __print(msg="", label=None, date=True, newline=True, ret=False):
        msg_aux = ""

        if date:
            str_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg_aux = msg_aux + f"<{str_time}> "

        if label is not None:
            str_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg_aux = msg_aux + f"[ {label} ] "

        msg_aux = msg_aux + f"{msg}"

        if newline:
            msg_aux = msg_aux + f"\n"

        print(f"{msg_aux}")

        if ret:
            return ret

    def print_raw(msg, tab="", ret=False):
        Log.__print(label=None, msg=msg, date=False, ret=ret)

    def line(newline=False, ret=False):
        Log.__print(
            msg="---------------------------------------------------",
            date=False,
            newline=False,
        )

    def printMessage(msg, newline=False):
        Log.line()
        Log.__print(msg=msg, newline=False)
        Log.line()
        if newline:
            print("")

    def info(msg, newline=True, ret=False):
        Log.__print(label="INFO", msg=msg, newline=newline, ret=ret)

    def err(msg, ret=False):
        Log.__print(label="ERROR", msg=msg, ret=ret)

    def debug(msg, ret=False):
        Log.__print(label="DEBUG", msg=msg, ret=ret)

    def warn(msg, ret=False):
        Log.__print(label="WARN", msg=msg, ret=ret)

    def ok(ret=False):
        Log.__print(label="OK", msg="", ret=ret)
