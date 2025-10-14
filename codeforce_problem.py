import sys

def expected_packets(N, A, B):
    E = [0.0]*(N+1)
    P = [0.0]*(N+1)  # prefix sums of E

    if A > 0:
        m = B - A + 1
        for r in range(1, N+1):
            L = max(0, r - B)
            U = r - A
            if U >= L:
                s = P[U] - (P[L-1] if L > 0 else 0.0)
            else:
                s = 0.0
            E[r] = 1.0 + s / m
            P[r] = P[r-1] + E[r]
    else:
        m = B + 1  # includes the zero-size packet
        for r in range(1, N+1):
            L = max(0, r - B)
            U = r - 1
            if U >= L:
                s = P[U] - (P[L-1] if L > 0 else 0.0)
            else:
                s = 0.0
            E[r] = (m/(m-1.0)) + s/(m-1.0)
            P[r] = P[r-1] + E[r]

    return E[N]

if __name__ == "__main__":
    N, A, B = map(int, sys.stdin.read().strip().split())
    ans = expected_packets(N, A, B)
    print("{:.10f}".format(ans))
