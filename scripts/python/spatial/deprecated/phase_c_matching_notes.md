# Phase C Matching Notes

Phase C uses an inclusive spatial matching workflow. Records are preserved when possible, quality issues are flagged, and downstream users can filter later. A record is excluded from spatial matching only when it lacks usable coordinates.

## Count definitions

`total site-feature matches` is the number of rows produced by the spatial join between coordinate-bearing ITRDB site records and the final buffered analysis polygons. One ITRDB code can appear more than once when it falls inside more than one feature polygon.

`matched ITRDB codes` is the number of unique matched `root_site_code_guess` values among those site-feature matches. This removes duplicate feature overlaps but still counts ITRDB code variants separately when the parser keeps them as distinct site codes.

`matched product records` is the number of rows in `outputs/inventories/itrdb_products.csv` whose `root_site_code_guess` belongs to the matched ITRDB codes. This is a file-discovery count, not a spatial-intersection count. It helps users find all available RWL, CRN, metadata, and related product files for matched codes.

`matched physical locations` is the number of parser-assisted physical-site groups represented by the matched ITRDB codes. This is the preferred scientific reporting unit because it collapses product/code variants that refer to the same physical sampling location.

## Relationship among counts

The accepted Phase C architecture is:

```text
site-feature matches
  -> matched ITRDB codes
      -> matched product records
          -> matched physical locations
```

These counts answer different questions and are not expected to be equal.

## Why 223 != 952 != 2958 is expected

`223` represents matched physical locations in the accepted aggregation. It is lower than the matched ITRDB-code count because several ITRDB code variants can refer to the same physical place.

`952` represents matched ITRDB codes. It is higher than the physical-location count because code variants and product-associated site codes are still represented before physical-site aggregation.

`2958` represents matched product records. It is higher than the matched-code count because one matched ITRDB code can have multiple product records, such as RWL files, CRN files, metadata files, species variants, or other inventory entries.

In short, `223 != 952 != 2958` is expected because Phase C reports physical places, matched site codes, and available product files as separate layers of evidence.

## Inclusive preservation policy

Phase C does not hide matched records merely because species, country, elevation, RWL availability, CRN availability, metadata, or text encoding are incomplete or imperfect. Those issues are carried forward as status, warning, and diagnostics fields.

Only records lacking usable coordinates are non-spatially-matchable. All other matched records are preserved so users can inspect quality and make project-specific filtering decisions later.

## User-facing reporting guidance

Use `matched physical locations` as the preferred scientific reporting unit.

Use `matched product records` to locate available ITRDB files associated with matched sites.
