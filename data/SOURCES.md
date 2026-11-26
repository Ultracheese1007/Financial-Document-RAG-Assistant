# Data provenance

The six records in `demo.json` are manually curated paraphrases of real published FY2025 Microsoft financial disclosures. They are **not** verbatim extracted passages. The full report is not distributed in this repository. Each record links to its original source. Amounts preserve the published units and reporting period; the CEO letter rounds consolidated revenue and operating income to billions.

- Microsoft Annual Report 2025, CEO letter: https://www.microsoft.com/investor/reports/ar25/ — consolidated revenue $281.7bn, operating income $128.5bn; annual increases 15% and 17%.
- Same report, management discussion and analysis, overview: cloud and AI infrastructure investments and their cost/margin implications; competition and macroeconomic/geopolitical conditions.
- Same report, income tax disclosures: income taxes paid net of refunds, $28.7bn, $23.4bn, $23.1bn (FY2025–FY2023).
- Microsoft FY2025 Form 10-K, Item 8, Income Statements, printed p.50: https://www.sec.gov/Archives/edgar/data/789019/000095017025100235/msft-20250630.htm — net income $101,832m (FY2025) and $88,136m (FY2024).

Full text-based Form 10-K PDF link: https://microsoft.gcs-web.com/static-files/9640e86d-a99f-4502-934d-1d3c883fa786

The PDF ingestion unit test creates a **synthetic ExampleCo PDF** locally. It is not a real company's report. The evaluation set is hand-authored against the curated demo facts and should not be used to claim general model accuracy.
