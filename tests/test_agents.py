import unittest
import numpy as np
from marl.agents.student import StudentAgent
from marl.agents.teacher import TeacherAgent

class TestAgents(unittest.TestCase):
    
    def setUp(self):
        # Create simple configs for testing
        self.state_dim = 10
        self.action_dim = 5
        self.student_config = {'learning_rate': 0.01, 'epsilon': 1.0}
        self.teacher_config = {'learning_rate': 0.01, 'epsilon': 0.5}
        
    def test_student_initialization(self):
        student = StudentAgent(self.state_dim, self.action_dim, self.student_config)
        self.assertEqual(student.state_dim, self.state_dim)
        self.assertEqual(student.action_dim, self.action_dim)
        self.assertEqual(student.config['epsilon'], 1.0)
        
    def test_student_act(self):
        student = StudentAgent(self.state_dim, self.action_dim, self.student_config)
        state = np.zeros(self.state_dim)
        valid_actions = [0, 1, 2]
        
        # With epsilon=1.0, actions should be random from valid_actions
        actions = [student.act(state, valid_actions) for _ in range(100)]
        for action in actions:
            self.assertIn(action, valid_actions)
            
    # More tests for Student and Teacher agents

if __name__ == '__main__':
    unittest.main()