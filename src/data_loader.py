import openpyxl
import os


def read_DAMEND_from_excel(file_path):
    wb = openpyxl.load_workbook(file_path, data_only=True)
    ws = wb.active
    row_idx = 1
    col_idx = 1
    found = False

    for row_idx in range(1, ws.max_row + 1):
        for col_idx in range(1, ws.max_column + 1):
            cell_value = ws.cell(row=row_idx, column=col_idx).value
            if cell_value and "DAMEND" in str(cell_value).upper():
                found = True
                break
        if found:
            break

    if not found:
        print("DAMEND non trovato, esco")
        exit()

    matrix_demand = []
    list_row = []
    row_idx += 1
    col_idx += 1 
    initial_col_idx = col_idx  # Store the initial column index for resetting
    while True:
        cell_value = ws.cell(row=row_idx, column=col_idx).value
        if isinstance(cell_value, int) and cell_value > 0:
            list_row.append(cell_value)
            col_idx += 1
        elif cell_value is None and col_idx != initial_col_idx:
            row_idx += 1
            col_idx = initial_col_idx  # Reset column index to the initial value
            matrix_demand.append(list_row)  # Add an empty row to the matrix
            list_row = []  # Reset the list for the next row
            continue  # Continue to the next row
        elif cell_value is None and col_idx == initial_col_idx:
            break  # Stop reading if the first cell of the row is empty
        else:
            exit(1)
    else:
        exit(1)

    return matrix_demand



def read_CAPACITY_from_excel(file_path):
    wb = openpyxl.load_workbook(file_path, data_only=True)
    ws = wb.active
    row_idx = 1
    col_idx = 1
    found = False

    for row_idx in range(1, ws.max_row + 1):
        cell_value = ws.cell(row=row_idx, column=col_idx).value
        if cell_value and "CAPACITY" in str(cell_value).upper():
            found = True
            break

    if not found:
        print("CAPACITY non trovata, esco")
        exit()

    matrix_capacity = []
    list_row = []
    row_idx += 1
    col_idx += 1 
    initial_col_idx = col_idx  # Store the initial column index for resetting
    while True:
        cell_value = ws.cell(row=row_idx, column=col_idx).value
        if isinstance(cell_value, int) and cell_value > 0:
            list_row.append(cell_value)
            col_idx += 1
        elif cell_value is None and col_idx != initial_col_idx:
            row_idx += 1
            col_idx = initial_col_idx  # Reset column index to the initial value
            matrix_capacity.append(list_row)  # Add an empty row to the matrix
            list_row = []  # Reset the list for the next row
            continue  # Continue to the next row
        elif cell_value is None and col_idx == initial_col_idx:
            break  # Stop reading if the first cell of the row is empty
        else:
            exit(1)
    else:
        exit(1)
    return matrix_capacity

def read_COST_from_excel(file_path):
    wb = openpyxl.load_workbook(file_path, data_only=True)
    ws = wb.active
    row_idx = 1
    col_idx = 1
    found = False

    for row_idx in range(1, ws.max_row + 1):
        for col_idx in range(1, ws.max_column + 1):
            cell_value = ws.cell(row=row_idx, column=col_idx).value
            if cell_value and "COST" in str(cell_value).upper():
                found = True
                break
        if found:
            break

    if not found:
        print("COST non trovato, esco")
        exit()

    print("COST trovato, il programma continua")    
    matrix_cost = []
    list_row = []
    row_idx += 1
    col_idx += 1 
    initial_col_idx = col_idx  # Store the initial column index for resetting
    while True:
        cell_value = ws.cell(row=row_idx, column=col_idx).value
        if isinstance(cell_value, int) and cell_value > 0:
            list_row.append(cell_value)
            col_idx += 1
        elif cell_value is None and col_idx != initial_col_idx:
            row_idx += 1
            col_idx = initial_col_idx  # Reset column index to the initial value
            matrix_cost.append(list_row)  # Add an empty row to the matrix
            list_row = []  # Reset the list for the next row
            continue  # Continue to the next row
        elif cell_value is None and col_idx == initial_col_idx:
            break  # Stop reading if the first cell of the row is empty
        else:
            exit(1)
    else:
        exit(1)
    return matrix_cost


def read_PRICE_from_excel(file_path):
    wb = openpyxl.load_workbook(file_path, data_only=True)
    ws = wb.active
    row_idx = 1
    col_idx = 1
    found = False

    for row_idx in range(1, ws.max_row + 1):
        for col_idx in range(1, ws.max_column + 1):
            cell_value = ws.cell(row=row_idx, column=col_idx).value
            if cell_value and "PRICE" in str(cell_value).upper():
                found = True
                break
        if found:
            break

    if not found:
        print("PRICE non trovato, esco")
        exit()

    matrix_price = []
    list_row = []
    row_idx += 1
    col_idx += 1 
    initial_col_idx = col_idx  # Store the initial column index for resetting
    while True:
        cell_value = ws.cell(row=row_idx, column=col_idx).value
        if isinstance(cell_value, int) and cell_value > 0:
            list_row.append(cell_value)
            col_idx += 1
        elif cell_value is None and col_idx != initial_col_idx:
            row_idx += 1
            col_idx = initial_col_idx  # Reset column index to the initial value
            matrix_price.append(list_row)  # Add an empty row to the matrix
            list_row = []  # Reset the list for the next row
            continue  # Continue to the next row
        elif cell_value is None and col_idx == initial_col_idx:
            break  # Stop reading if the first cell of the row is empty
        else:
            exit(1)
    else:
        exit(1)
    return matrix_price

def read_REVENUE_from_excel(file_path):
    wb = openpyxl.load_workbook(file_path, data_only=True)
    ws = wb.active
    row_idx = 1
    col_idx = 1
    found = False

    for row_idx in range(1, ws.max_row + 1):
        for col_idx in range(1, ws.max_column + 1):
            cell_value = ws.cell(row=row_idx, column=col_idx).value
            if cell_value and "REVENUE" in str(cell_value).upper():
                found = True
                break
        if found:
            break

    if not found:
        print("REVENUE non trovato, esco")
        exit()

    matrix_revenue = []
    list_row = []
    row_idx += 1
    col_idx += 1 
    initial_col_idx = col_idx  # Store the initial column index for resetting
    while True:
        cell_value = ws.cell(row=row_idx, column=col_idx).value
        if isinstance(cell_value, int) and cell_value > 0:
            list_row.append(cell_value)
            col_idx += 1
        elif cell_value is None and col_idx != initial_col_idx:
            row_idx += 1
            col_idx = initial_col_idx  # Reset column index to the initial value
            matrix_revenue.append(list_row)  # Add an empty row to the matrix
            list_row = []  # Reset the list for the next row
            continue  # Continue to the next row
        elif cell_value is None and col_idx == initial_col_idx:
            break  # Stop reading if the first cell of the row is empty
        else:
            exit(1)
    else:
        exit(1)
    return matrix_revenue



if __name__ == "__main__":
    # File path
    file_path = os.path.join("..", "quarantine_hotel_instances", "1.xlsx")
    
    matrix = read_DAMEND_from_excel(file_path)
    print ("DAMEND Matrix:")
    for row in matrix:
        print(row)

    print("\n\n")

    matrix_capacity = read_CAPACITY_from_excel(file_path)
    print ("CAPACITY Matrix:")
    for row in matrix_capacity:
        print(row)

    print("\n\n")

    matrix_cost = read_COST_from_excel(file_path)
    print ("COST Matrix:")
    for row in matrix_cost:
        print(row)

    print("\n\n")

    matrix_price = read_PRICE_from_excel(file_path)
    print ("PRICE Matrix:")
    for row in matrix_price:
        print(row)

    print("\n\n")

    matrix_revenue = read_REVENUE_from_excel(file_path)
    print ("REVENUE Matrix:")
    for row in matrix_revenue:
        print(row)

    print("\n\n")