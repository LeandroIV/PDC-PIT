# Simple PowerShell script to convert Markdown files to Word documents
# This script simply copies the content to a Word document with minimal formatting

# Function to convert Markdown to Word (simple version)
function Convert-MDtoWord-Simple {
    param (
        [string]$MarkdownFile,
        [string]$OutputFile
    )

    Write-Host "Converting $MarkdownFile to $OutputFile..."

    # Create Word application
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false

    # Create a new document
    $doc = $word.Documents.Add()
    $selection = $word.Selection

    # Read the markdown file
    $content = Get-Content -Path $MarkdownFile -Raw

    # Simply paste the content
    $selection.TypeText($content)

    # Save the document
    $doc.SaveAs([ref]$OutputFile, [ref]16)  # 16 = wdFormatDocumentDefault
    $doc.Close()
    $word.Quit()

    # Release COM objects
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($selection) | Out-Null
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($doc) | Out-Null
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()

    Write-Host "Conversion complete: $OutputFile"
}

# Get the current directory
$currentDir = Get-Location

# List of Markdown files to convert
$markdownFiles = @(
    "GPU_Programming_Report.md",
    "GPU_Programming_Presentation.md",
    "GPU_Programming_Code_Examples.md",
    "GPU_Programming_Visual_Assets.md",
    "README.md"
)

# Convert each file
foreach ($file in $markdownFiles) {
    $inputPath = Join-Path -Path $currentDir -ChildPath $file
    $outputPath = Join-Path -Path $currentDir -ChildPath ($file -replace '\.md$', '.docx')
    
    if (Test-Path -Path $inputPath) {
        Convert-MDtoWord-Simple -MarkdownFile $inputPath -OutputFile $outputPath
    } else {
        Write-Host "File not found: $inputPath"
    }
}

Write-Host "All conversions completed!"
