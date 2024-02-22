from background_population.bg_population import BackgroundPopulation
from environments.repeated_prisoners import RepeatedPrisonersDilemmaEnv

if __name__ == '__main__':


    bg = BackgroundPopulation(RepeatedPrisonersDilemmaEnv(2))

    print(bg.build_randomly(4))