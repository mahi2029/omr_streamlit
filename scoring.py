from bubble_detection import extract_answers

def score_sheet(sheet_path, answer_key, debug=False):
    """
    Compare detected answers with the given answer key.
    answer_key should be a dictionary like:
    {1: 2, 2: 0, 3: 1}  where values are indices (0=A, 1=B, 2=C, 3=D)
    """
    detected = extract_answers(sheet_path, debug=debug)
    score = 0
    total = len(answer_key)
    results = {}

    for q, correct_ans in answer_key.items():
        given_ans = detected.get(q, None)

        if given_ans == correct_ans:
            results[q] = ("Correct", given_ans, correct_ans)
            score += 1
        elif given_ans is None:
            results[q] = ("No Answer", None, correct_ans)
        else:
            results[q] = ("Wrong", given_ans, correct_ans)

    return score, total, results


if _name_ == "_main_":
    # Example Answer Key: {Q#: correct option index}
    # 0 = A, 1 = B, 2 = C, 3 = D
    answer_key = {
        1: 2,  # Q1 correct = C
        2: 0,  # Q2 correct = A
        3: 1,  # Q3 correct = B
        4: 3   # Q4 correct = D
    }

    sheet_path = "C:/Users/mahis/OneDrive/Desktop/a4397defd26be38b74d96a98a34689d4.jpg"
    score, total, results = score_sheet(sheet_path, answer_key, debug=True)

    print("\n--- Evaluation Report ---")
    for q, (status, given, correct) in results.items():
        given_str = chr(65+given) if given is not None else "None"
        correct_str = chr(65+correct)
        print(f"Q{q}: {status} (Your: {given_str}, Correct: {correct_str})")

    print(f"\nFinal Score: {score}/{total}")