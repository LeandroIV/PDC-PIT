# PowerShell script to convert Markdown files to Word documents
# This script uses the Microsoft Word COM object to create Word documents

# Function to convert Markdown to Word
function Convert-MarkdownToWord {
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

    # Process the content
    # Split the content by lines
    $lines = $content -split "`r`n|`r|`n"

    foreach ($line in $lines) {
        # Handle headings
        if ($line -match '^#{1,6}\s+(.+)$') {
            $level = $matches[0].IndexOf(' ')
            $text = $matches[1]
            
            # Apply heading style based on level
            switch ($level) {
                1 { $selection.Style = $word.ActiveDocument.Styles.Item("Heading 1") }
                2 { $selection.Style = $word.ActiveDocument.Styles.Item("Heading 2") }
                3 { $selection.Style = $word.ActiveDocument.Styles.Item("Heading 3") }
                4 { $selection.Style = $word.ActiveDocument.Styles.Item("Heading 4") }
                5 { $selection.Style = $word.ActiveDocument.Styles.Item("Heading 5") }
                6 { $selection.Style = $word.ActiveDocument.Styles.Item("Heading 6") }
            }
            
            $selection.TypeText($text)
            $selection.TypeParagraph()
        }
        # Handle code blocks
        elseif ($line -match '^```') {
            # Skip the opening ```
            $selection.TypeParagraph()
            $selection.Style = $word.ActiveDocument.Styles.Item("No Spacing")
            $selection.Font.Name = "Courier New"
            $selection.Font.Size = 10
            
            # Continue reading until closing ```
            $i = [array]::IndexOf($lines, $line) + 1
            while ($i -lt $lines.Count -and $lines[$i] -notmatch '^```') {
                $selection.TypeText($lines[$i])
                $selection.TypeParagraph()
                $i++
            }
            
            # Reset style
            $selection.Style = $word.ActiveDocument.Styles.Item("Normal")
            $selection.TypeParagraph()
        }
        # Handle bullet points
        elseif ($line -match '^\s*[\*\-\+]\s+(.+)$') {
            $text = $matches[1]
            $selection.Style = $word.ActiveDocument.Styles.Item("List Bullet")
            $selection.TypeText($text)
            $selection.TypeParagraph()
        }
        # Handle numbered lists
        elseif ($line -match '^\s*\d+\.\s+(.+)$') {
            $text = $matches[1]
            $selection.Style = $word.ActiveDocument.Styles.Item("List Number")
            $selection.TypeText($text)
            $selection.TypeParagraph()
        }
        # Handle tables
        elseif ($line -match '^\|.+\|$') {
            # This is a table row
            $cells = $line -split '\|' | Where-Object { $_ -ne '' }
            
            # If this is the first row of the table, create a table
            if (-not $inTable) {
                $tableRows = 1
                $tableCols = $cells.Count
                $table = $doc.Tables.Add($selection.Range, $tableRows, $tableCols)
                $inTable = $true
                
                # Fill the first row
                for ($col = 0; $col -lt $cells.Count; $col++) {
                    $table.Cell(1, $col + 1).Range.Text = $cells[$col].Trim()
                }
            }
            # If this is a separator row (---|---), skip it
            elseif ($line -match '^\|(\s*[\-:]+\s*\|)+\s*$') {
                continue
            }
            # Otherwise, add a new row to the table
            else {
                $table.Rows.Add()
                $tableRows++
                
                # Fill the new row
                for ($col = 0; $col -lt $cells.Count; $col++) {
                    if ($col + 1 -le $tableCols) {
                        $table.Cell($tableRows, $col + 1).Range.Text = $cells[$col].Trim()
                    }
                }
            }
        }
        # If we were in a table but this line is not a table row, end the table
        elseif ($inTable -and $line -notmatch '^\|.+\|$') {
            $inTable = $false
            $selection.EndOf(6) | Out-Null  # End of story
            $selection.TypeParagraph()
        }
        # Handle regular text
        elseif ($line -ne '') {
            $selection.Style = $word.ActiveDocument.Styles.Item("Normal")
            
            # Handle bold and italic
            $line = $line -replace '\*\*(.+?)\*\*', '$1'  # Bold
            $line = $line -replace '\*(.+?)\*', '$1'      # Italic
            $line = $line -replace '\_\_(.+?)\_\_', '$1'  # Bold
            $line = $line -replace '\_(.+?)\_', '$1'      # Italic
            
            # Handle links
            $line = $line -replace '\[(.+?)\]\((.+?)\)', '$1'
            
            $selection.TypeText($line)
            $selection.TypeParagraph()
        }
        else {
            # Empty line
            $selection.TypeParagraph()
        }
    }

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
        Convert-MarkdownToWord -MarkdownFile $inputPath -OutputFile $outputPath
    } else {
        Write-Host "File not found: $inputPath"
    }
}

Write-Host "All conversions completed!"
