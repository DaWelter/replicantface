import random
from pathlib import Path
import copy
import os


class PersistentShuffledCycle:
    def __init__(self, items : list[str|int], path : Path | str):
        if isinstance(path, str):
            path = Path(path)
        self._type = type(next(iter(items)))
        self._items = items
        self._path = path
        if not self._try_restore_iter():
            self._todo = []

    def _try_restore_iter(self) -> bool:
        try:
            with self._path.open('r') as f:
                todo = [ self._type(s.strip()) for s in f.readlines() ]
        except IOError as e:
            return False
        if not set(todo).issubset(set(self._items)):
            print(f"Stored items aren't a subset of the total item list: {todo} from {self._path}")
            return False
        self._todo = todo
        return True

    def _restart(self) -> bool:
        todo = copy.copy(self._items)
        random.shuffle(todo)
        self._todo = todo

    def _save_bg_iter(self):
        with self._path.open('w') as f:
            f.write('\n'.join(str(x) for x in self._todo))

    def next(self):
        try:
            return self._todo.pop()
        except IndexError:
            self._restart()
        assert self._todo
        return self._todo.pop()

    def save(self):
        self._save_bg_iter()
    

if __name__ == '__main__':
    # Test
    os.unlink('/tmp/foo.txt')
    cycle = PersistentShuffledCycle(list(range(3)), '/tmp/foo.txt')
    out = [ cycle.next() for _ in range(6) ]
    assert sorted(out) == [0,0,1,1,2,2], f"Got {out}"
    assert out != [0,0,1,1,2,2], f"Got {out}" # Unlikely due to shuffling but not impossible.
    a = cycle.next()
    cycle.save()
    cycle = PersistentShuffledCycle(list(range(3)), '/tmp/foo.txt')
    assert (b := cycle.next()) != a
    assert (c := cycle.next()) != a
    assert sorted([a,b,c]) == [0,1,2], f"Got {a}, {b}, {c}"