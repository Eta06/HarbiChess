"""Pure human-origin role/source wiring; teacher endpoints fail closed."""
from admission import SEARCH_SHA, SEEDS, VARIANT
from teacher_admission import pin, read


def validate_wiring(q):
    if q['target_variant'] != VARIANT or q['match_seeds'] != list(SEEDS) or 'teachers' in q:
        raise ValueError('human-origin explicit target and both seeds; no teacher endpoint alias')
    for name in ('original_mixed_value', 'E8_value_helper', 'prior_helper', 'binary',
                 'original_search', 'collection_audit_set'):
        pin(q[name])
    if q['original_search']['sha256'] != SEARCH_SHA:
        raise ValueError('same exact original512 all-root search for every arm')
    for seed in SEEDS:
        row = q['models'][str(seed)]
        parent = read(q['parents'][str(seed)]['admission_seal'])
        if (set(row) != {'learned', 'parent', 'e8'}
                or row['parent'] != parent['parent_candidate']
                or row['learned'] != q['children'][str(seed)]['candidate']
                or row['e8']['sha256'] !=
                'e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03'):
            raise ValueError('exact current admitted parent/child/E8 endpoints')
    return q
