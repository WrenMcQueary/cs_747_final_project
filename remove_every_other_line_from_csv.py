if __name__ == "__main__":
    with open("NEW_my_solution.csv", "w") as f_new:
        with open("my_solution.csv", "r") as f_old:
            for i, line in enumerate(f_old):
                if i % 2 == 0:
                    f_new.write(line)
