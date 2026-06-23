from rag_agent.agent import run_agent


DEFAULT_MESSAGE = "Great Tesla with latest news about Elon Musk"


def print_execution_trace(response) -> None:
    print("🎩 Alfred's Response:")
    print("\nExecution trace:")
    for message in response["messages"]:
        message.pretty_print()


def main() -> None:
    response = run_agent(DEFAULT_MESSAGE)
    print_execution_trace(response)


if __name__ == "__main__":
    main()
