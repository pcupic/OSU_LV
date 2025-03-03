if __name__ == "__main__":
    numbers = []
    
    while True:
        user_input = input("Enter a number (or type 'Done' to finish): ")
        
        if user_input.lower() == "done":
            break
        
        try:
            number = float(user_input)
            numbers.append(number)
        except ValueError:
            print("Invalid input, please enter a valid number.")
    
    if numbers:
        print(f"Total numbers entered: {len(numbers)}")
        print(f"Average value: {sum(numbers) / len(numbers):.2f}")
        print(f"Minimum value: {min(numbers)}")
        print(f"Maximum value: {max(numbers)}")
        numbers.sort()
        print(f"Sorted list: {numbers}")
    else:
        print("No valid numbers were entered.")
