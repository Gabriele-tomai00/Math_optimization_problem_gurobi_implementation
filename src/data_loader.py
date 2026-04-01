import openpyxl
import os


def find_label(ws, label):
    for r in range(1, ws.max_row + 1):
        for c in range(1, ws.max_column + 1):
            val = ws.cell(row=r, column=c).value
            if val and label in str(val).upper():
                return r, c
    raise ValueError(f"{label} not found")


def read_block(ws, start_row, start_col, is_vector=False):
    data = []
    row = start_row + 1
    col_start = start_col + 1

    while True:
        col = col_start
        row_data = []

        cell_value = ws.cell(row=row, column=col).value

        # Stop if first cell is empty
        if cell_value is None:
            break

        while True:
            cell_value = ws.cell(row=row, column=col).value

            if isinstance(cell_value, int) and cell_value > 0:
                row_data.append(cell_value)
                col += 1
            else:
                break

        if is_vector:
            data.append(row_data[0])
        else:
            data.append(row_data)

        row += 1

    return data

def read_penalty(ws):
    found_capacity = False
    found_empty = False

    for row_idx in range(1, ws.max_row + 1):
        cell_value = ws.cell(row=row_idx, column=1).value

        # Find CAPACITY
        if cell_value and "CAPACITY" in str(cell_value).upper():
            found_capacity = True
            continue

        # After CAPACITY
        if found_capacity:
            if cell_value is None:
                found_empty = True
                continue

            if found_empty and isinstance(cell_value, int) and cell_value % 100 == 0:
                return cell_value

    raise ValueError("Penalty not found")

def read_from_excel(ws, label, is_vector=False):

    r, c = find_label(ws, label)
    return read_block(ws, r, c, is_vector)

def pretty_print(name, data):
    print(f"{name}:")
    
    if isinstance(data[0], list):  # matrix
        for row in data:
            print("  ", row)
    else:  # vector
        for i, val in enumerate(data, 1):
            print("  ", val)    


def get_data_from_file_excel(file_path):
    wb = openpyxl.load_workbook(file_path, data_only=True)
    ws = wb.active
    all_instances = {}

    for ws in wb.worksheets:
        sheet_name = ws.title

        # Keep only sheets with numeric names
        if not sheet_name.isdigit():
            continue

        sheet_name = ws.title
        print(f"Processing sheet: {sheet_name}")
        damend = read_from_excel(ws, "DAMEND")
        capacity = read_from_excel(ws, "CAPACITY")
        cost = read_from_excel(ws, "COST")
        price = read_from_excel(ws, "PRICE")
        revenue = read_from_excel(ws, "REVENUE", is_vector=True)
        penalty = read_penalty(ws)

        all_instances[sheet_name] = {
            "damend": damend,
            "capacity": capacity,
            "cost": cost,
            "price": price,
            "revenue": revenue,
            "penalty": penalty
        }
    return all_instances

if __name__ == "__main__":

    all_files_data = {}

    for i in range(1, 17):
        file_name = f"{i}.xlsx"
        file_path = os.path.join("..", "quarantine_hotel_instances", file_name)

        if not os.path.exists(file_path):
            print(f"File {file_name} non trovato, salto")
            continue

        print(f"\n=== Processing {file_name} ===")

        instances = get_data_from_file_excel(file_path)
        all_files_data[file_name] = instances

        for sheet_name, data in instances.items():
            print(f"\n--- {file_name} | Sheet {sheet_name} ---")
            pretty_print("DAMEND", data["damend"])
            pretty_print("CAPACITY", data["capacity"])
            pretty_print("COST", data["cost"])
            pretty_print("PRICE", data["price"])
            pretty_print("REVENUE", data["revenue"])
            pretty_print("PENALTY", [data["penalty"]])
