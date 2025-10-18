from nim import train, play

def main():
    # Number of training games
    n = 1000

    # Train AI
    print()
    for i in range(1, n + 1):
        print(f"Playing training game {i}")
    ai = train(n)
    print("Done training")
    print()

    # Play against the trained AI
    play(ai)


if __name__ == "__main__":
    main()
