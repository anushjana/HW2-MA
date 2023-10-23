from Bandits import EpsilonGreedy, ThompsonSampling


def main():

    trials = 2000
    mean = 0.4
    e = EpsilonGreedy(mean)
    result_e = e.experiment(trials)
    e.plot1(result_e)

    mean_t = 6
    t = ThompsonSampling(mean_t)
    result_t = t.experiment(trials)
    t.plot1(result_t)
    print(t.report(result_t, result_e))

if __name__ == '__main__':
    main()



