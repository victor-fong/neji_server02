from threading import Condition

class RecentQueue:
    def __init__(self, max_size):

        self.max_size = max_size
        self.queue = []
        # self.lock = Lock()
        self.cond = Condition()

    def enqueue(self, element):
        with self.cond:
            self.queue.append(element)
            if len(self.queue) > self.max_size:
                self.queue = self.queue[1:]
            self.cond.notify()

    def dequeue(self):
        with self.cond:
            while len(self.queue) == 0:
                self.cond.wait()
            result = self.queue[0]
            self.queue = self.queue[1:]
            return result

if __name__ == "__main__":
    queue = RecentQueue(2)
    queue.enqueue("1")
    queue.enqueue("2")
    queue.enqueue("3")
    first_result = queue.dequeue()
    print(f"First Result {first_result}")
    assert first_result == "2"
    second_result = queue.dequeue()
    print(f"Second Result {second_result}")
    assert second_result == "3"
