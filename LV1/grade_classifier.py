def get_grade(score):
    if score >= 0.9:
        return "A"
    elif score >= 0.8:
        return "B"
    elif score >= 0.7:
        return "C"
    elif score >= 0.6:
        return "D"
    else:
        return "F"

if __name__ == "__main__":
    try:
        score = float(input("Enter a score between 0.0 and 1.0: "))
        if 0.0 <= score <= 1.0:
            print(f"Grade: {get_grade(score)}")
        else:
            print("Error: Score must be between 0.0 and 1.0")
    except ValueError:
        print("Error: Invalid input, please enter a numerical value.")
