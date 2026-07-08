class Letter:
    """Base class for letters in the environment. Default behavior is static."""
    def __init__(self, char: str):
        self.char = char
        self.x = -1
        self.y = -1

    def step(self, letter_env):
        # Default: static letter, do nothing
        pass

    def reset(self):
        # Default: no reset needed for static letters
        pass

class MovingLetter(Letter):
    """Letter that moves with constant velocity, stops when blocked."""
    def __init__(self, char: str, dx: int = 0, dy: int = 0):
        super().__init__(char)
        self.dx = dx
        self.dy = dy
        self.start_dx = dx
        self.start_dy = dy

    def step(self, letter_env):
        grid_size = letter_env.grid_size
        new_x = (self.x + self.dx) % grid_size
        new_y = (self.y + self.dy) % grid_size
        # Only move if new position is empty (not in map)
        if (new_x, new_y) not in letter_env.map:
            self.x = new_x
            self.y = new_y

    def reset(self):
        self.dx = self.start_dx
        self.dy = self.start_dy

class RandomMovingLetter(Letter):
    """Letter that randomly moves to an available cardinal neighbor each step."""
    def step(self, letter_env):
        grid_size = letter_env.grid_size
        actions = letter_env.actions
        candidates = []
        for di, dj in actions:
            new_x = (self.x + di) % grid_size
            new_y = (self.y + dj) % grid_size
            if (new_x, new_y) not in letter_env.map:
                candidates.append((new_x, new_y))
        if candidates:
            new_x, new_y = letter_env.rng.choice(candidates)
            self.x = new_x
            self.y = new_y

class BouncingLetter(MovingLetter):
    """Letter that bounces (reverses velocity) when hitting obstacles."""
    def step(self, letter_env):
        grid_size = letter_env.grid_size
        new_x = (self.x + self.dx) % grid_size
        new_y = (self.y + self.dy) % grid_size
        # If new position is blocked, bounce and try again
        if (new_x, new_y) in letter_env.map:
            self.dx *= -1
            self.dy *= -1
            new_x = (self.x + self.dx) % grid_size
            new_y = (self.y + self.dy) % grid_size
        # Move if new position is empty
        if (new_x, new_y) not in letter_env.map:
            self.x = new_x
            self.y = new_y