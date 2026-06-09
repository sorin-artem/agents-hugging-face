import asyncio

from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step, Event


class ProcessingEvent(Event):
    intermediate_result: str


class MyWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> ProcessingEvent:
        return ProcessingEvent(intermediate_result="Hello, world!")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        # Use the intermediate result
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)


async def main() -> None:
    workflow = MyWorkflow(timeout=10, verbose=False)
    result = await workflow.run()
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
