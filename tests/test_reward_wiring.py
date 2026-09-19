"""Every term a config names must actually be exported by the mdp package.

`mdp/__init__.py` re-exports by an explicit list, not by `import *`, and the
configs refer to terms as `mdp.<name>`. A term added to a config but not to that
list fails only at env construction time, inside IsaacSim, after the GPU has
been claimed -- and the mdp package cannot even be imported without the Sim app
(`omni`), so the CPU suite cannot catch it by importing. This parses both files
instead, so the gap is caught in a second.

The check covers every `mdp.<name>` in env_cfg, not just RewTerm callbacks:
DoneTerm (the goal termination), EventTerm and rewards all resolve the same way.
Names the package does not define are assumed to come from the isaaclab
wildcard import, which cannot be enumerated statically.
"""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / 'source/booster_train/booster_train/tasks/manager_based/kick_amp/mdp'
ENV_CFG = ROOT / ('source/booster_train/booster_train/tasks/manager_based/'
                  'kick_amp/robots/k1/kick_amp/env_cfg.py')
REWARDS_SRC = MDP / 'rewards.py'
COMMANDS_SRC = MDP / 'commands.py'


def exported_names():
    """Everything mdp/__init__.py re-exports explicitly (not the `import *` lines)."""
    tree = ast.parse((MDP / '__init__.py').read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module != '*':
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def configured_rewards():
    """Names referenced as mdp.<name> from a rewards position in env_cfg."""
    tree = ast.parse(ENV_CFG.read_text())
    names = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'RewTerm'):
            continue
        # RewTerm(func=mdp.x, weight=...) passes the reward as a keyword, so
        # args[0] is empty; accept either form rather than silently seeing none.
        func = node.args[0] if node.args else next(
            (kw.value for kw in node.keywords if kw.arg == 'func'), None)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) \
                and func.value.id == 'mdp':
            names.add(func.attr)
    return names


def defined_in_package():
    """Names our mdp package defines itself: functions, classes, CONSTANTS."""
    names = set()
    for path in MDP.glob('*.py'):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name) and t.id.isupper())
    return names


def referenced_mdp_names():
    """Every mdp.<name> attribute accessed anywhere in env_cfg."""
    tree = ast.parse(ENV_CFG.read_text())
    return {node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == 'mdp'}


class MdpWiring(unittest.TestCase):
    """The stronger contract: any name the config uses must be re-exported."""

    def test_every_referenced_package_name_is_exported(self):
        missing = sorted((referenced_mdp_names() & defined_in_package()) - exported_names())
        self.assertEqual(missing, [], f'env_cfg uses mdp terms the package does not export: {missing}')

    def test_parser_finds_the_references_and_the_export_list(self):
        referenced = referenced_mdp_names()
        self.assertGreater(len(referenced), 10, f'only found {sorted(referenced)}')
        self.assertIn('pos_still', referenced)
        # The generalization's point: DoneTerm callbacks must be seen too, not
        # just RewardCfg entries.
        self.assertIn('time_out', referenced, 'DoneTerm references must be parsed')
        exports = exported_names()
        self.assertGreater(len(exports), 20, f'only exported {sorted(exports)}')
        self.assertIn('ball_in_goal', exports)


class RewardWiring(unittest.TestCase):
    def test_config_only_names_exported_rewards(self):
        missing = sorted(configured_rewards() - exported_names())
        self.assertEqual(missing, [], f'RewardCfg names terms the mdp package does not export: {missing}')

    def test_parser_actually_finds_the_reward_terms(self):
        """Guard the guard: a silent regex/AST drift would make the test vacuous."""
        configured = configured_rewards()
        self.assertGreater(len(configured), 10, f'only found {sorted(configured)}')
        self.assertIn('boundary_distance', configured)
        self.assertIn('pos_still', configured)


if __name__ == '__main__':
    unittest.main()


class ConstantImports(unittest.TestCase):
    """A term must import every module constant its body uses.

    ball_lateral_speed referenced GOAL_X without importing it and every test
    passed, because the function-lifting harness supplies the names itself. This
    checks the file's own imports instead, which is what the simulator loads.
    """

    IMPORTS_FROM = 'commands'

    def imported_names(self):
        tree = ast.parse(REWARDS_SRC.read_text())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == self.IMPORTS_FROM:
                names.update(alias.asname or alias.name for alias in node.names)
        return names

    def used_constants(self):
        """UPPERCASE names referenced by rewards.py that commands.py defines."""
        defined = set()
        tree = ast.parse(COMMANDS_SRC.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                defined.update(t.id for t in node.targets if isinstance(t, ast.Name) and t.id.isupper())
        used = {n.id for n in ast.walk(ast.parse(REWARDS_SRC.read_text()))
                if isinstance(n, ast.Name) and n.id.isupper()}
        return used & defined

    def test_every_used_constant_is_imported(self):
        missing = sorted(self.used_constants() - self.imported_names())
        self.assertEqual(missing, [], f'rewards.py uses {missing} without importing them')
        self.assertIn('GOAL_X', self.used_constants(), 'guard the guard: parsing must find the constants')
