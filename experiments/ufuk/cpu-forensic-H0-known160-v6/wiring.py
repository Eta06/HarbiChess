"""Pure typed protocol presence/role/source checks before native admission."""

from admission import SEARCH_SHA, SEEDS, VARIANTS
from teacher_admission import pin


def validate_wiring(q):
    if q['target_variant'] not in {'own-q-v2', *VARIANTS}:
        raise ValueError('unknown target variant')
    if q['match_seeds'] != list(SEEDS):
        raise ValueError('fixed both-seed development endpoint')
    for name in ('original_mixed_value', 'E8_value_helper', 'prior_helper', 'binary',
                 'original_search', 'collection_audit_set'):
        pin(q[name])
    if q['original_search']['sha256'] != SEARCH_SHA:
        raise ValueError('same exact original512/q2 all-root search')
    for seed in SEEDS:
        roles = q['models'][str(seed)]
        if q['target_variant'] == 'forensic-own-v5-h0':
            parent = q['parents'][str(seed)]['candidate']
        else:
            parent = q['teachers'][str(seed)]['candidate']
        if (set(roles) != {'learned', 'parent', 'e8'}
                or roles['parent'] != parent
                or roles['learned'] != q['children'][str(seed)]['candidate']
                or roles['e8']['sha256'] !=
                'e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03'):
            raise ValueError('exact fixed child/exact teacher ancestor/E8 roles')
    return q
