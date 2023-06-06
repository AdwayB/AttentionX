import eeek
import tips

if __name__ == '__main__':
    eeek.run()
    ret_tips = tips.analyze_telemetry()
    print(ret_tips)