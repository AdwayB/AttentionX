import eeek
import tips


def return_tips():
    eeek.run()
    ret_tips = tips.analyze_telemetry()
    print(ret_tips)

    return ret_tips


if __name__ == '__main__':
    print(return_tips())
