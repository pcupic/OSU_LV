def calculate_earnings(hours, rate):
    return hours * rate

if __name__ == "__main__":
    hours = float(input("Enter the number of working hours: "))
    rate = float(input("Enter the hourly rate: "))
    earnings = calculate_earnings(hours, rate)
    print(f"Total earnings: {earnings} euros")
