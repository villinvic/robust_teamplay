# from background_population.bg_population import BackgroundPopulation
# from environments.matrix_form.repeated_prisoners import RepeatedPrisonersDilemmaEnv
#
# if __name__ == '__main__':
#
#
#     bg = BackgroundPopulation(RepeatedPrisonersDilemmaEnv(2))
#
#     print(bg.build_randomly(4))
import time

from rich.progress import Progress


progress = Progress()


class Job():

    def __init__(self):
        pass


    def __call__(self, *args, **kwargs):


        task = progress.add_task("[red]Working...", total=100)
        for i in range(100):
            time.sleep(0.1)



for i in range(10):
