for n in range(1, 151):
    filename = f"Contract{n}.sol"
    with open(filename, 'w') as file:
        # Write content to the file if needed
        file.write("")
    print(f"File '{filename}' created successfully.")