def longSubseg(strS, changeK):
    length = len(strS)
    left = 0
    max_int = 0
    count = 0
    ways = 0

    # First pass to find the maximum length of consecutive 1s
    for current in range(length):
        if strS[current] == '0':
            count += 1
        while count > changeK:
            if strS[left] == '0':
                count -= 1
            left += 1
        max_int = max(max_int, current - left + 1)

    # Reset variables for the second pass
    left = 0
    count = 0

    # Second pass to count the number of ways to achieve the maximum length
    for current in range(length):
        if strS[current] == '0':
            count += 1
        while count > changeK:
            if strS[left] == '0':
                count -= 1
            left += 1
        if (current - left + 1) == max_int:
            # Check if this is a new way to achieve the max length
            if current == length - 1 or strS[current + 1] == '0':
                ways += 1

    return ways


def main():
    # Input for strS
    strS = input().strip()

    # Input for changeK
    changeK = int(input().strip())

    result = longSubseg(strS, changeK)
    print(result)


if __name__ == "__main__":
    main()
