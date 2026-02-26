def is_finite_decimal(a, b):
    # Reduce the fraction to lowest terms
    from math import gcd
    g = gcd(a, b)
    b = abs(b // g)
    # Remove all 2s from denominator
    while b % 2 == 0:
        b //= 2
    # Remove all 5s from denominator
    while b % 5 == 0:
        b //= 5
    # If what's left is 1, it's a terminating decimal
    return b == 1

# Examples:

time_step = 0.002
frequency = int(1/time_step)


for i in range(1000, 20000):
    if is_finite_decimal(frequency, i):
        print(f"Perfect decimal found: {frequency}/{i} = ", frequency/i)
