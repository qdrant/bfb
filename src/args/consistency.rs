//! CLI value types for write ordering and read consistency, with conversions
//! to their qdrant-client counterparts.

use std::{fmt, str};

use qdrant_client::qdrant;

#[derive(Copy, Clone, Debug)]
pub enum WriteOrdering {
    Weak,
    Medium,
    Strong,
}

impl fmt::Display for WriteOrdering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let str = match self {
            Self::Weak => "Weak",
            Self::Medium => "Medium",
            Self::Strong => "Strong",
        };

        str.fmt(f)
    }
}

impl str::FromStr for WriteOrdering {
    type Err = anyhow::Error;

    fn from_str(str: &str) -> Result<Self, Self::Err> {
        match str {
            "Weak" => Ok(Self::Weak),
            "Medium" => Ok(Self::Medium),
            "Strong" => Ok(Self::Strong),
            _ => Err(anyhow::format_err!(
                "invalid WriteOrdering value {str}, \
                 valid values are Weak, Medium or Strong"
            )),
        }
    }
}

impl From<WriteOrdering> for qdrant::WriteOrdering {
    fn from(ordering: WriteOrdering) -> Self {
        qdrant::WriteOrdering {
            r#type: ordering.into(),
        }
    }
}

impl From<WriteOrdering> for i32 {
    fn from(ordering: WriteOrdering) -> Self {
        qdrant::WriteOrderingType::from(ordering) as _
    }
}

impl From<WriteOrdering> for qdrant::WriteOrderingType {
    fn from(ordering: WriteOrdering) -> Self {
        match ordering {
            WriteOrdering::Weak => Self::Weak,
            WriteOrdering::Medium => Self::Medium,
            WriteOrdering::Strong => Self::Strong,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ReadConsistency {
    Type(ReadConsistencyType),
    Factor(u64),
}

impl fmt::Display for ReadConsistency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Type(consistency) => consistency.fmt(f),
            Self::Factor(factor) => factor.fmt(f),
        }
    }
}

impl str::FromStr for ReadConsistency {
    type Err = anyhow::Error;

    fn from_str(str: &str) -> Result<Self, Self::Err> {
        if let Ok(consistency) = str.parse() {
            return Ok(Self::Type(consistency));
        }

        if let Ok(factor) = str.parse() {
            return Ok(Self::Factor(factor));
        }

        Err(anyhow::format_err!(
            "invalid ReadConsistency value {str}, \
             valid values are All, Majority, Quorum or a positive integer number"
        ))
    }
}

impl From<ReadConsistency> for qdrant::read_consistency::Value {
    fn from(consistency: ReadConsistency) -> Self {
        match consistency {
            ReadConsistency::Type(consistency) => consistency.into(),
            ReadConsistency::Factor(factor) => qdrant::read_consistency::Value::Factor(factor),
        }
    }
}

impl From<ReadConsistency> for qdrant::ReadConsistency {
    fn from(consistency: ReadConsistency) -> Self {
        let consistency = match consistency {
            ReadConsistency::Type(consistency) => consistency.into(),
            ReadConsistency::Factor(factor) => qdrant::read_consistency::Value::Factor(factor),
        };

        qdrant::ReadConsistency {
            value: consistency.into(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ReadConsistencyType {
    All,
    Majority,
    Quorum,
}

impl fmt::Display for ReadConsistencyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let str = match self {
            Self::All => "All",
            Self::Majority => "Majority",
            Self::Quorum => "Quorum",
        };

        str.fmt(f)
    }
}

impl str::FromStr for ReadConsistencyType {
    type Err = anyhow::Error;

    fn from_str(str: &str) -> Result<Self, Self::Err> {
        match str {
            "All" => Ok(Self::All),
            "Majority" => Ok(Self::Majority),
            "Quorum" => Ok(Self::Quorum),
            _ => Err(anyhow::format_err!(
                "invalid ReadConsistencyType value {str}, \
                 valid values are All, Majority or Quorum"
            )),
        }
    }
}

impl From<ReadConsistencyType> for qdrant::read_consistency::Value {
    fn from(consistency: ReadConsistencyType) -> Self {
        qdrant::read_consistency::Value::Type(consistency.into())
    }
}

impl From<ReadConsistencyType> for i32 {
    fn from(consistency: ReadConsistencyType) -> Self {
        qdrant::ReadConsistencyType::from(consistency) as _
    }
}

impl From<ReadConsistencyType> for qdrant::ReadConsistencyType {
    fn from(consistency: ReadConsistencyType) -> Self {
        match consistency {
            ReadConsistencyType::All => qdrant::ReadConsistencyType::All,
            ReadConsistencyType::Majority => qdrant::ReadConsistencyType::Majority,
            ReadConsistencyType::Quorum => qdrant::ReadConsistencyType::Quorum,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_ordering_from_str() {
        assert!(matches!(
            "Weak".parse::<WriteOrdering>().unwrap(),
            WriteOrdering::Weak
        ));
        assert!(matches!(
            "Medium".parse::<WriteOrdering>().unwrap(),
            WriteOrdering::Medium
        ));
        assert!(matches!(
            "Strong".parse::<WriteOrdering>().unwrap(),
            WriteOrdering::Strong
        ));
        assert!("weak".parse::<WriteOrdering>().is_err());
        assert!("".parse::<WriteOrdering>().is_err());
    }

    #[test]
    fn test_write_ordering_display_roundtrip() {
        for ordering in [
            WriteOrdering::Weak,
            WriteOrdering::Medium,
            WriteOrdering::Strong,
        ] {
            let s = ordering.to_string();
            let parsed: WriteOrdering = s.parse().unwrap();
            assert_eq!(ordering.to_string(), parsed.to_string());
        }
    }

    #[test]
    fn test_read_consistency_type_from_str() {
        assert!(matches!(
            "All".parse::<ReadConsistencyType>().unwrap(),
            ReadConsistencyType::All
        ));
        assert!(matches!(
            "Majority".parse::<ReadConsistencyType>().unwrap(),
            ReadConsistencyType::Majority
        ));
        assert!(matches!(
            "Quorum".parse::<ReadConsistencyType>().unwrap(),
            ReadConsistencyType::Quorum
        ));
        assert!("all".parse::<ReadConsistencyType>().is_err());
        assert!("".parse::<ReadConsistencyType>().is_err());
    }

    #[test]
    fn test_read_consistency_type_display_roundtrip() {
        for consistency in [
            ReadConsistencyType::All,
            ReadConsistencyType::Majority,
            ReadConsistencyType::Quorum,
        ] {
            let s = consistency.to_string();
            let parsed: ReadConsistencyType = s.parse().unwrap();
            assert_eq!(consistency.to_string(), parsed.to_string());
        }
    }

    #[test]
    fn test_read_consistency_from_str_with_factor() {
        let parsed: ReadConsistency = "3".parse().unwrap();
        assert!(matches!(parsed, ReadConsistency::Factor(3)));
    }

    #[test]
    fn test_read_consistency_from_str_with_type() {
        let parsed: ReadConsistency = "All".parse().unwrap();
        assert!(matches!(
            parsed,
            ReadConsistency::Type(ReadConsistencyType::All)
        ));
    }
}
