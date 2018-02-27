import csv

# Generates CSV files of training and testing data TODO: write to files


def generate_test_and_train():
    with open('credit_card_default_data.csv') as RawData:
        total_men = 0
        total_women = 0
        default_men = 0
        default_women = 0
        data = csv.reader(RawData)
        # Skips the first two lines
        next(data)
        next(data)
        for row in data:
            # Updates totals for men
            if int(row[2]) == 1:
                total_men += 1
                if int(row[24]) == 1:
                    default_men += 1

            # Updates totals for women
            if int(row[2]) == 2:
                total_women += 1
                if int(row[24]) == 1:
                    default_women += 1

        print("Male default percentage: ", (default_men / total_men))
        print("Female default percentage: ", (default_women / total_women))
        # print("Total men ", total_men)
        # print("Total women: ", total_women)


if __name__ == "__main__":
    generate_test_and_train()
