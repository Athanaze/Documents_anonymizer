using System;
using System.Text;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Platform.Storage;
using UglyToad.PdfPig;
using UglyToad.PdfPig.DocumentLayoutAnalysis.TextExtractor;
using System.Text.RegularExpressions;

namespace AvaloniaApplication2;

public class PdfLoader
{
    
    string RedactNames(string input)
    {
        // Regular expression for matching names (flexible pattern)
        const string pattern = @"\b([A-Z]\w+)\b";  

        return Regex.Replace(input, pattern, (Match m) => m.Groups[1].Value[0] + "___");
    }

    string RedactEmailAddresses(string input)
    {
        // Regular expression to match common email formats
        var pattern = @"(\w)[\w\.-]+@[\w\.-]+";

        return Regex.Replace(input, pattern, (Match m) => m.Groups[1].Value + "___@___");
    }

    string ProcessStr(string input)
    {
        return RedactNames(RedactEmailAddresses(input));
    }
    
    public static async Task<string> PickPdfFileAsync(Window parent)
    {
        var dialog = new OpenFileDialog
        {
            Title = "Select a PDF file",
            
        };

        var result = await dialog.ShowAsync(parent);

        if (result != null && result.Length > 0)
        {
            return result[0];
        }

        return null;
    }

    public static string toText(string filePath)
    {
        using (var pdf = PdfDocument.Open(filePath))
        {
            var textBuilder = new StringBuilder();
            foreach (var page in pdf.GetPages())
            {
                textBuilder.AppendLine(ContentOrderTextExtractor.GetText(page));
            }
            return textBuilder.ToString();
        }
    }
}