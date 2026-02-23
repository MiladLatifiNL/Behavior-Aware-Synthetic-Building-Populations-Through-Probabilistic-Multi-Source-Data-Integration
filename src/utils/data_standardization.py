"""
Data standardization utilities for PUMS Enrichment Pipeline.

This module provides functions to standardize names, addresses, and other
fields for consistent processing and improved matching accuracy.
"""

import re
import unicodedata
from typing import Dict, Tuple, Optional, List
import jellyfish
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Common name prefixes and suffixes
NAME_PREFIXES = {
    'MR', 'MRS', 'MS', 'MISS', 'DR', 'REV', 'FR', 'SR', 'JR', 
    'PROF', 'SIR', 'LADY', 'LORD', 'DAME', 'CAPT', 'COL', 'MAJ', 'GEN'
}

NAME_SUFFIXES = {
    'JR', 'SR', 'II', 'III', 'IV', 'V', 'ESQ', 'PHD', 'MD', 'DDS', 
    'DVM', 'LLD', 'JD', 'RN', 'DO', 'DPM'
}

# Common name abbreviations and their expansions
NAME_ABBREVIATIONS = {
    'WM': 'WILLIAM',
    'CHAS': 'CHARLES',
    'JAS': 'JAMES',
    'THOS': 'THOMAS',
    'ROBT': 'ROBERT',
    'JOS': 'JOSEPH',
    'BENJ': 'BENJAMIN',
    'SAM': 'SAMUEL',
    'ALEX': 'ALEXANDER',
    'ELIZ': 'ELIZABETH',
    'MARG': 'MARGARET',
    'CATH': 'CATHERINE',
    'BARB': 'BARBARA',
    'VICT': 'VICTORIA',
    'JR': 'JUNIOR',
    'SR': 'SENIOR'
}

# Street type standardizations
STREET_TYPES = {
    'STREET': 'ST', 'STR': 'ST', 'STRT': 'ST',
    'AVENUE': 'AVE', 'AVEN': 'AVE', 'AVENU': 'AVE', 'AVN': 'AVE', 'AVNUE': 'AVE',
    'ROAD': 'RD', 'RDS': 'RD',
    'DRIVE': 'DR', 'DRV': 'DR', 'DRIV': 'DR', 'DRV': 'DR',
    'LANE': 'LN', 'LNE': 'LN',
    'COURT': 'CT', 'CRT': 'CT',
    'PLACE': 'PL', 'PLC': 'PL',
    'BOULEVARD': 'BLVD', 'BOUL': 'BLVD', 'BOULV': 'BLVD',
    'HIGHWAY': 'HWY', 'HWAY': 'HWY', 'HIWAY': 'HWY',
    'PARKWAY': 'PKWY', 'PARKWY': 'PKWY', 'PKY': 'PKWY',
    'SQUARE': 'SQ', 'SQR': 'SQ', 'SQRE': 'SQ',
    'CIRCLE': 'CIR', 'CIRC': 'CIR', 'CIRCL': 'CIR', 'CRCL': 'CIR', 'CRCLE': 'CIR',
    'TRAIL': 'TRL', 'TRAILS': 'TRL', 'TRL': 'TRL',
    'WAY': 'WAY', 'WY': 'WAY',
    'TERRACE': 'TER', 'TERR': 'TER', 'TERRASSE': 'TER'
}

# Directional standardizations
DIRECTIONS = {
    'NORTH': 'N', 'NO': 'N', 'NTH': 'N',
    'SOUTH': 'S', 'SO': 'S', 'STH': 'S',
    'EAST': 'E', 'EST': 'E',
    'WEST': 'W', 'WST': 'W',
    'NORTHEAST': 'NE', 'NORTH-EAST': 'NE', 'N.E.': 'NE',
    'NORTHWEST': 'NW', 'NORTH-WEST': 'NW', 'N.W.': 'NW',
    'SOUTHEAST': 'SE', 'SOUTH-EAST': 'SE', 'S.E.': 'SE',
    'SOUTHWEST': 'SW', 'SOUTH-WEST': 'SW', 'S.W.': 'SW'
}

# Unit designations
UNIT_DESIGNATIONS = {
    'APARTMENT': 'APT', 'APT.': 'APT', 'APPT': 'APT', 
    'SUITE': 'STE', 'STE.': 'STE', 'SUIT': 'STE',
    'UNIT': 'UNIT', 'UN': 'UNIT',
    'ROOM': 'RM', 'RM.': 'RM',
    'FLOOR': 'FL', 'FL.': 'FL', 'FLR': 'FL',
    'BUILDING': 'BLDG', 'BLDG.': 'BLDG', 'BLD': 'BLDG'
}


def remove_accents(text: str) -> str:
    """Remove accents from Unicode characters."""
    if not text:
        return text
    nfkd_form = unicodedata.normalize('NFKD', text)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])


def standardize_names(name: str) -> Dict[str, str]:
    """
    Standardize a person's name.
    
    Args:
        name: Full name string
        
    Returns:
        Dictionary with standardized name components:
        - full_name: Cleaned full name
        - first_name: First name
        - middle_name: Middle name/initial
        - last_name: Last name
        - prefix: Title/prefix (Mr., Dr., etc.)
        - suffix: Suffix (Jr., III, etc.)
        - soundex_first: Soundex code for first name
        - soundex_last: Soundex code for last name
        - nysiis_first: NYSIIS code for first name
        - nysiis_last: NYSIIS code for last name
    """
    if not name or not isinstance(name, str):
        return {
            'full_name': '',
            'first_name': '',
            'middle_name': '',
            'last_name': '',
            'prefix': '',
            'suffix': '',
            'soundex_first': '',
            'soundex_last': '',
            'nysiis_first': '',
            'nysiis_last': ''
        }
    
    # Convert to uppercase and remove accents
    name = remove_accents(name.upper())
    
    # Remove multiple spaces and trim
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Remove special characters except hyphens and apostrophes
    name = re.sub(r"[^A-Z\s\-']", '', name)
    
    # Extract prefix
    prefix = ''
    parts = name.split()
    if parts and parts[0].replace('.', '') in NAME_PREFIXES:
        prefix = parts[0].replace('.', '')
        parts = parts[1:]
    
    # Extract suffix
    suffix = ''
    if parts and parts[-1].replace('.', '') in NAME_SUFFIXES:
        suffix = parts[-1].replace('.', '')
        parts = parts[:-1]
    
    # Expand abbreviations
    expanded_parts = []
    for part in parts:
        clean_part = part.replace('.', '')
        if clean_part in NAME_ABBREVIATIONS:
            expanded_parts.append(NAME_ABBREVIATIONS[clean_part])
        else:
            expanded_parts.append(clean_part)
    
    # Parse name components
    first_name = ''
    middle_name = ''
    last_name = ''
    
    if len(expanded_parts) == 1:
        last_name = expanded_parts[0]
    elif len(expanded_parts) == 2:
        first_name = expanded_parts[0]
        last_name = expanded_parts[1]
    elif len(expanded_parts) >= 3:
        first_name = expanded_parts[0]
        middle_name = ' '.join(expanded_parts[1:-1])
        last_name = expanded_parts[-1]
    
    # Generate phonetic codes
    soundex_first = ''
    soundex_last = ''
    nysiis_first = ''
    nysiis_last = ''
    
    if first_name:
        try:
            soundex_first = jellyfish.soundex(first_name)
            nysiis_first = jellyfish.nysiis(first_name)
        except:
            pass
    
    if last_name:
        try:
            soundex_last = jellyfish.soundex(last_name)
            nysiis_last = jellyfish.nysiis(last_name)
        except:
            pass
    
    # Reconstruct full name
    full_name_parts = []
    if prefix:
        full_name_parts.append(prefix)
    if first_name:
        full_name_parts.append(first_name)
    if middle_name:
        full_name_parts.append(middle_name)
    if last_name:
        full_name_parts.append(last_name)
    if suffix:
        full_name_parts.append(suffix)
    
    full_name = ' '.join(full_name_parts)
    
    return {
        'full_name': full_name,
        'first_name': first_name,
        'middle_name': middle_name,
        'last_name': last_name,
        'prefix': prefix,
        'suffix': suffix,
        'soundex_first': soundex_first,
        'soundex_last': soundex_last,
        'nysiis_first': nysiis_first,
        'nysiis_last': nysiis_last
    }


def standardize_addresses(address: str) -> Dict[str, str]:
    """
    Standardize an address string.
    
    Args:
        address: Full address string
        
    Returns:
        Dictionary with standardized address components:
        - full_address: Cleaned full address
        - street_number: House/building number
        - street_name: Street name
        - street_type: Street type (ST, AVE, etc.)
        - unit_type: Unit type (APT, STE, etc.)
        - unit_number: Unit number
        - direction: Directional indicator (N, S, E, W, etc.)
        - po_box: PO Box number if applicable
    """
    if not address or not isinstance(address, str):
        return {
            'full_address': '',
            'street_number': '',
            'street_name': '',
            'street_type': '',
            'unit_type': '',
            'unit_number': '',
            'direction': '',
            'po_box': ''
        }
    
    # Convert to uppercase and clean
    address = address.upper()
    address = re.sub(r'\s+', ' ', address).strip()
    
    # Check for PO Box
    po_box_match = re.search(r'P\.?O\.?\s*BOX\s*(\d+)', address)
    if po_box_match:
        return {
            'full_address': f'PO BOX {po_box_match.group(1)}',
            'street_number': '',
            'street_name': '',
            'street_type': '',
            'unit_type': '',
            'unit_number': '',
            'direction': '',
            'po_box': po_box_match.group(1)
        }
    
    # Initialize components
    street_number = ''
    street_name = ''
    street_type = ''
    unit_type = ''
    unit_number = ''
    direction = ''
    
    # Extract unit information
    unit_match = re.search(r'(' + '|'.join(UNIT_DESIGNATIONS.keys()) + r')\s*#?\s*(\w+)', address, re.I)
    if unit_match:
        unit_type_raw = unit_match.group(1).upper()
        unit_type = UNIT_DESIGNATIONS.get(unit_type_raw, unit_type_raw)
        unit_number = unit_match.group(2)
        # Remove unit info from address for further parsing
        address = address[:unit_match.start()] + address[unit_match.end():]
    
    # Split address into parts
    parts = address.split()
    
    # Extract street number
    if parts and re.match(r'^\d+[A-Z]?$', parts[0]):
        street_number = parts[0]
        parts = parts[1:]
    
    # Extract directional prefix
    if parts and parts[0] in DIRECTIONS:
        direction = DIRECTIONS[parts[0]]
        parts = parts[1:]
    elif parts and parts[0] in DIRECTIONS.values():
        direction = parts[0]
        parts = parts[1:]
    
    # Extract street type from end
    if parts:
        last_part = parts[-1]
        if last_part in STREET_TYPES:
            street_type = STREET_TYPES[last_part]
            parts = parts[:-1]
        elif last_part in STREET_TYPES.values():
            street_type = last_part
            parts = parts[:-1]
    
    # Extract directional suffix if not already found
    if not direction and parts:
        last_part = parts[-1]
        if last_part in DIRECTIONS:
            direction = DIRECTIONS[last_part]
            parts = parts[:-1]
        elif last_part in DIRECTIONS.values():
            direction = last_part
            parts = parts[:-1]
    
    # Remaining parts form the street name
    street_name = ' '.join(parts)
    
    # Reconstruct standardized address
    address_parts = []
    if street_number:
        address_parts.append(street_number)
    if direction and street_name:  # Pre-directional
        address_parts.append(direction)
    if street_name:
        address_parts.append(street_name)
    if not street_name and direction:  # Post-directional
        address_parts.append(direction)
    if street_type:
        address_parts.append(street_type)
    
    full_address = ' '.join(address_parts)
    
    if unit_type and unit_number:
        full_address += f' {unit_type} {unit_number}'
    
    return {
        'full_address': full_address,
        'street_number': street_number,
        'street_name': street_name,
        'street_type': street_type,
        'unit_type': unit_type,
        'unit_number': unit_number,
        'direction': direction,
        'po_box': ''
    }


def parse_name_components(name: str) -> Tuple[str, str, str]:
    """
    Parse name into first, middle, and last components.
    
    Args:
        name: Full name string
        
    Returns:
        Tuple of (first_name, middle_name, last_name)
    """
    result = standardize_names(name)
    return result['first_name'], result['middle_name'], result['last_name']


def standardize_field(value: any, field_type: str = 'general') -> str:
    """
    General field standardization.
    
    Args:
        value: Value to standardize
        field_type: Type of field ('name', 'address', 'general')
        
    Returns:
        Standardized string value
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ''
    
    # Convert to string
    value = str(value).strip()
    
    if field_type == 'name':
        result = standardize_names(value)
        return result['full_name']
    elif field_type == 'address':
        result = standardize_addresses(value)
        return result['full_address']
    else:
        # General standardization
        value = value.upper()
        value = re.sub(r'\s+', ' ', value)
        value = re.sub(r'[^\w\s\-]', '', value)
        return value.strip()


def create_blocking_key(record: Dict[str, any], key_fields: List[str]) -> str:
    """
    Create a blocking key from specified fields.
    
    Args:
        record: Record dictionary
        key_fields: List of fields to use for blocking key
        
    Returns:
        Blocking key string
    """
    key_parts = []
    
    for field in key_fields:
        if field in record:
            value = str(record[field])
            
            # Special handling for names - use first letter of last name
            if 'name' in field.lower() and 'last' in field.lower():
                name_parts = standardize_names(value)
                if name_parts['last_name']:
                    key_parts.append(name_parts['last_name'][0])
            # Special handling for geographic codes
            elif field.lower() in ['state', 'puma', 'county']:
                key_parts.append(value[:3])  # Use first 3 characters
            else:
                # General field - use first character
                clean_value = standardize_field(value)
                if clean_value:
                    key_parts.append(clean_value[0])
    
    return '|'.join(key_parts)


if __name__ == "__main__":
    # Test name standardization
    test_names = [
        "Dr. John A. Smith, Jr.",
        "Mary Jane O'Brien",
        "José María García-López",
        "WM JOHNSON",
        "Smith",
        "ELIZ BROWN-JONES III"
    ]
    
    print("Name Standardization Tests:")
    print("-" * 80)
    for name in test_names:
        result = standardize_names(name)
        print(f"Original: {name}")
        print(f"Standardized: {result['full_name']}")
        print(f"Components: First='{result['first_name']}', Middle='{result['middle_name']}', Last='{result['last_name']}'")
        print(f"Phonetic: Soundex={result['soundex_last']}, NYSIIS={result['nysiis_last']}")
        print()
    
    # Test address standardization
    test_addresses = [
        "123 West Main Street, Apt. 45B",
        "456 N BROADWAY AVE SUITE 100",
        "P.O. Box 789",
        "789 Martin Luther King Jr Boulevard",
        "321 5TH AVENUE NORTH"
    ]
    
    print("\nAddress Standardization Tests:")
    print("-" * 80)
    for address in test_addresses:
        result = standardize_addresses(address)
        print(f"Original: {address}")
        print(f"Standardized: {result['full_address']}")
        print(f"Components: Number='{result['street_number']}', Name='{result['street_name']}', Type='{result['street_type']}'")
        if result['unit_type']:
            print(f"Unit: {result['unit_type']} {result['unit_number']}")
        print()