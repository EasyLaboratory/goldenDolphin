from loader.loaderPipeline import hisar_loader_pipeline
from conf import LOGO
from rich.console import Console
from loader.RML2018a import RML2018a

if __name__ == "__main__":
    console = Console()
    console.print(LOGO, style="yellow")
    RUNNING_FLAG = True
    while RUNNING_FLAG:
        console.print("please choose the dateset you want to process:")
        console.print("1:hisar 2019 dataset")
        console.print("2:RML 2018 dataset")
        choice = console.input("please select:")
        if choice == '1':
            console.print("select your batch size,default value is 10000")
            value = console.input("please set the value:")
            hisar_loader_pipeline(int(value))
        elif choice == '2':
            RML2018 = RML2018a(console)
            # display information about the dataset
            RML2018.describe()
            RML2018.filter_snr(2, 25)
        RUNNING_FLAG = False

