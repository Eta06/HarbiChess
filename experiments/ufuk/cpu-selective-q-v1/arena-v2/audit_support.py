"""Independent charged extension receipt validation, no inference or search."""


def validate_extensions(row, role):
    expected = ('new-selective-Q-learned-or-zero' if role in ['learned', 'newzero']
                else 'original-prior-or-E8-search')
    if row['controller_kind'] != expected:
        raise ValueError('wrong controller for role')
    extensions = row['q_extension_receipts']
    if expected == 'original-prior-or-E8-search' and extensions:
        raise ValueError('original search must not have new extensions')
    if any(set(e) != {'nodes', 'status', 'checked'} or type(e['nodes']) is not int
           or not 0 <= e['nodes'] <= 64 or e['status'] not in ['completed', 'censored']
           or type(e['checked']) is not bool for e in extensions):
        raise ValueError('exact bounded extension receipt')
    if sum(e['nodes'] for e in extensions) > row['nodes']:
        raise ValueError('extension nodes exceed global charged budget')
