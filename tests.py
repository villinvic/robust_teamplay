# from background_population.bg_population import BackgroundPopulation
# from environments.matrix_form.repeated_prisoners import RepeatedPrisonersDilemmaEnv
#
# if __name__ == '__main__':
#
#
#     bg = BackgroundPopulation(RepeatedPrisonersDilemmaEnv(2))
#
#     print(bg.build_randomly(4))
import yaml
with open("test.YAML", "r") as f:
    d = yaml.safe_load(f)


print(d)
