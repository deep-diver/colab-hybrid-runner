print("\n=== [Cell 2] Processing data using variables from Cell 1 ===")
# Uses dataset_name and numbers defined in Cell 1
squared_numbers = [x ** 2 for x in numbers]
total_sum = sum(squared_numbers)

print(f"Processing target: '{dataset_name}'")
print(f"Squared Numbers: {squared_numbers}")
print(f"Total Sum: {total_sum}")
