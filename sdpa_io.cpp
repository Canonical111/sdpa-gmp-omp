/* -------------------------------------------------------------

This file is a component of SDPA
Copyright (C) 2004 SDPA Project

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA

------------------------------------------------------------- */

/* MODIFIED from upstream (GPLv2 2a notice), 2026-08-03: lower-bound validation of input indices; checked fgets in header reader. See git log. */
/* MODIFIED from upstream (GPLv2 2a notice), 2026-08-04: every gmp_fscanf/fscanf conversion is checked; truncated sparse records and initial-point target values are diagnosed. See git log. */
#define DIMACS_PRINT 0
#define MESSAGEBUFFER 512

/* MODIFIED from upstream (GPLv2 2a notice), 2026-08-04: initialise the SOCP_sp_* locals; they were passed to C.initialize() while never assigned. See git log. */
/* MODIFIED from upstream (GPLv2 2a notice), 2026-08-06: sparse records are read one LINE at a time and must carry exactly five fields; mDIM/nBLOCK/bLOCKsTRUCT are range-checked before they drive allocations; the sparse initial-point reader bounds the block index and maps an LP block through blockNumber[] as the data reader always did; readReal no longer requires a leading delimiter. Diagnostics carry the line number. See git log. */
#include <sdpa_io.h>
#include <vector>
#include <algorithm>
#include <cctype>
#include <climits>
#include <string>

namespace sdpa {

namespace {

// gmp_fscanf's return value was ignored at nearly every call site, so a malformed
// token left the previous contents of the destination in place and the run
// continued on data the file never contained. Every conversion now goes through
// these wrappers.
//
// The format string USED to be "%*[^0-9+-]%Fe". That is a scanset, and a scanset
// must match AT LEAST ONE character or the whole gmp_fscanf fails before %Fe
// ever runs. The first thing read from an initial-point file is its y vector, at
// offset 0, so an .ini-s file whose first character is a digit or a sign was
// REJECTED -- a false rejection of a valid file. example1.ini-s survived only
// because it happens to begin with a space; delete that one space and dd and qd
// still solve the file while gmp alone refuses it. The token reader below has no
// such requirement, and it matches what dd and qd already do.
//
// The conversion itself stays native: the token is validated as a decimal
// literal and then converted by gmp's own %Fe into the mpf_class at its full
// working precision. Nothing goes through strtod.

int readNumericToken(FILE *fpData, std::string &token) {
    token.clear();
    int c;
    do {
        c = fgetc(fpData);
        if (c == EOF) {
            return EOF;
        }
    } while (!std::isdigit(static_cast<unsigned char>(c)) && c != '+' && c != '-' && c != '.');

    do {
        token.push_back(static_cast<char>(c));
        c = fgetc(fpData);
    } while (c != EOF && c != ',' && c != '}' && !std::isspace(static_cast<unsigned char>(c)));
    if (c != EOF) {
        ungetc(c, fpData);
    }
    return 1;
}

// Is `token` a decimal literal: [+-]? digits with at most one point, and an
// optional exponent that fits in long long? mpf has no mantissa-length limit, so
// unlike dd/qd there is nothing to rewrite -- this is validation only, and it
// exists so that a malformed field is reported with its own line number rather
// than silently leaving the destination unchanged.
bool isDecimalLiteral(const std::string &token) {
    size_t pos = 0;
    if (pos < token.size() && (token[pos] == '+' || token[pos] == '-')) {
        ++pos;
    }
    size_t digitCount = 0;
    bool sawPoint = false;
    while (pos < token.size() && token[pos] != 'e' && token[pos] != 'E') {
        const char c = token[pos++];
        if (c == '.') {
            if (sawPoint) {
                return false;
            }
            sawPoint = true;
        } else if (std::isdigit(static_cast<unsigned char>(c))) {
            ++digitCount;
        } else {
            return false;
        }
    }
    if (digitCount == 0) {
        return false;
    }
    if (pos < token.size()) {
        ++pos; // the 'e' or 'E'
        if (pos < token.size() && (token[pos] == '+' || token[pos] == '-')) {
            ++pos;
        }
        if (pos == token.size()) {
            return false;
        }
        long long exponent = 0;
        while (pos < token.size()) {
            const char c = token[pos++];
            if (!std::isdigit(static_cast<unsigned char>(c))) {
                return false;
            }
            const int digit = c - '0';
            if (exponent > (LLONG_MAX - digit) / 10) {
                return false;
            }
            exponent = exponent * 10 + digit;
        }
    }
    return true;
}

enum ConvertStatus { CONVERT_OK = 0, CONVERT_MALFORMED = 1 };

ConvertStatus convertReal(const std::string &token, mpf_class &value) {
    if (!isDecimalLiteral(token)) {
        return CONVERT_MALFORMED;
    }
    if (gmp_sscanf(token.c_str(), "%Fe", value.get_mpf_t()) != 1) {
        return CONVERT_MALFORMED;
    }
    return CONVERT_OK;
}

// Clip a token for a diagnostic so one corrupt line cannot print megabytes.
std::string clipForMessage(const std::string &s) {
    return s.size() > 80 ? s.substr(0, 80) + "..." : s;
}

int readReal(FILE *fpData, mpf_class &value) {
    std::string token;
    if (readNumericToken(fpData, token) == EOF) {
        return EOF;
    }
    return convertReal(token, value) == CONVERT_OK ? 1 : 0;
}

void requireReal(FILE *fpData, mpf_class &value, const char *record, int a, int b, int c, int d) {
    if (readReal(fpData, value) <= 0) {
        fprintf(stderr, "SDPA input: missing or malformed value -- expected a number for %s (%d, %d, %d, %d)\n", record, a, b, c, d);
        rError("io::read missing or malformed value in input");
    }
}

void requireReal(FILE *fpData, mpf_class &value, const char *record, int index, int count) {
    if (readReal(fpData, value) <= 0) {
        fprintf(stderr, "SDPA input: missing or malformed value -- expected a number for %s, entry %d of %d\n", record, index, count);
        rError("io::read missing or malformed value in input");
    }
}

// Echo an offending header line without its terminator, so the diagnostic stays
// on one line.
void reportBadLine(const char *what, const char *line) {
    size_t len = 0;
    while (line[len] != '\0' && line[len] != '\n' && line[len] != '\r') {
        ++len;
    }
    fprintf(stderr, "%s: '%.*s'\n", what, static_cast<int>(len), line);
}

// ---------------------------------------------------------------------------
// Line-bounded sparse record reader.
//
// The SDPA sparse record grammar is one record per line. SDPLIB/README.md,
// "SDPA sparse format" -> "File Format", item 6:
//
//     6. The remaining lines of the file contain entries in the constraint
//        matrices, with one entry per line.  The format for each line is
//            <matno> <blkno> <i> <j> <entry>
//
// All 92 shipped SDPLIB problems obey it exactly: 3,285,833 record lines, zero
// records spanning a line break and zero records carrying a sixth field.
//
// The previous reader obtained each of the five fields with a separate
// whitespace-skipping conversion, so it could not see a line boundary at all. A
// four-field record therefore took its fifth field from the FOLLOWING line and
// every later field shifted by one. That alone would usually be caught by an
// index range check -- but "%d" applied to a decimal value token such as "1.0"
// consumes the "1" and leaves ".0" in the stream, so the borrowed field is
// repaid out of the next value and the stream RE-SYNCHRONISES after exactly one
// record. The file then reaches EOF cleanly, a different problem is solved, and
// the run reports pdOPT with exit status 0. Deleting the row index from line 500
// of SDPLIB's theta1.dat-s does precisely this: objValPrimal 22.9583 against a
// published optimum of 23.0, exit 0, in all three forks.
//
// Reading a whole line and requiring exactly five fields on it closes that. The
// added checks the old reader had (its EOF tests) only ever fired at end of
// file, which is why the existing CI -- which truncates at EOF -- never saw it.
// ---------------------------------------------------------------------------

// A record line longer than this is refused instead of grown without bound. The
// widest record any shipped problem produces is a few dozen bytes; the cap
// exists only so that a corrupt file cannot drive an unbounded allocation.
const size_t SDPA_MAX_RECORD_LINE = 1u << 20;

// Field separators. Whitespace is the documented separator. ',', '{' and '}' are
// also treated as separators because the reader being replaced skipped them
// silently (its "%*[^0-9+-]" scanset consumed any non-numeric run), and this
// reader must not reject an input the old one accepted.
inline bool isFieldDelim(char c) {
    return std::isspace(static_cast<unsigned char>(c)) != 0 || c == ',' || c == '{' || c == '}';
}

// Read one physical line. Returns false only at end of file with nothing read.
// `overlong` reports that the line exceeded SDPA_MAX_RECORD_LINE; the excess is
// dropped rather than buffered, so the caller can diagnose without the file
// dictating the allocation.
bool readPhysicalLine(FILE *fp, std::string &line, bool &overlong) {
    line.clear();
    overlong = false;
    bool anything = false;
    int c;
    while ((c = fgetc(fp)) != EOF) {
        anything = true;
        if (c == '\n') {
            return true;
        }
        if (line.size() < SDPA_MAX_RECORD_LINE) {
            line.push_back(static_cast<char>(c));
        } else {
            overlong = true;
        }
    }
    return anything;
}

// Number of complete lines before byte offset `position`. The record sections
// start a few lines into the file (just after the c vector, or the y vector for
// an initial point), so this pass is negligible beside reading the records --
// and it is what lets a record diagnostic name a line of the FILE. Not one
// diagnostic in these readers carried a line number before; every message quoted
// field VALUES, which after a shift come from neighbouring lines and actively
// misdirect.
// Also reports, via `atLineStart`, whether `position` sits at the beginning of a
// line. It usually does NOT: the data reader starts just after the last number of
// the c vector and the initial-point reader just after the last number of the y
// vector, both mid-line. The first "line" such a reader sees is the tail of that
// header line, which is not a record and must not be validated as one -- the
// reader being replaced skipped it, because "%*[^0-9+-]" consumed any non-numeric
// run. Validating it would REJECT a valid file whose c vector is followed by
// trailing text, which is a worse defect than the one being fixed.
long linesBefore(FILE *fp, long position, bool *atLineStart) {
    *atLineStart = true;
    const long saved = ftell(fp);
    if (saved < 0 || position < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        return 0;
    }
    long lines = 0;
    long remaining = position;
    char buf[8192];
    char last = '\n';
    while (remaining > 0) {
        const long chunk = remaining < static_cast<long>(sizeof(buf)) ? remaining : static_cast<long>(sizeof(buf));
        const size_t got = fread(buf, 1, static_cast<size_t>(chunk), fp);
        if (got == 0) {
            break;
        }
        for (size_t t = 0; t < got; ++t) {
            if (buf[t] == '\n') {
                ++lines;
            }
        }
        last = buf[got - 1];
        remaining -= static_cast<long>(got);
    }
    *atLineStart = (last == '\n');
    fseek(fp, saved, SEEK_SET);
    return lines;
}

// Strict decimal integer. The old reader used "%d", which accepts the leading
// "1" of "1.0" and leaves ".0" behind -- the mechanism that let a truncated
// record re-synchronise silently. Every one of the corpus's 3,285,833 record
// lines carries a plain integer in fields 1-4, so nothing valid needs that
// tolerance.
bool parseFieldInt(const std::string &token, int &out) {
    if (token.empty()) {
        return false;
    }
    size_t pos = 0;
    bool negative = false;
    if (token[0] == '+' || token[0] == '-') {
        negative = token[0] == '-';
        pos = 1;
    }
    if (pos >= token.size()) {
        return false;
    }
    long long acc = 0;
    for (; pos < token.size(); ++pos) {
        if (!std::isdigit(static_cast<unsigned char>(token[pos]))) {
            return false;
        }
        acc = acc * 10 + (token[pos] - '0');
        if (acc > 2147483648LL) { // beyond |INT_MIN|; stops before long long overflows
            return false;
        }
    }
    if (negative) {
        acc = -acc;
    }
    if (acc > INT_MAX || acc < INT_MIN) {
        return false;
    }
    out = static_cast<int>(acc);
    return true;
}

// Reads sparse records one line at a time. `fileKind` and `firstFieldName` only
// shape the diagnostics: the data file's first field is <matrix>, the initial
// point file's is <target>.
class SparseRecordReader {
  public:
    SparseRecordReader(FILE *fp, const char *fileKind, const char *firstFieldName, long position)
        : fp_(fp), fileKind_(fileKind), firstFieldName_(firstFieldName), lineNo_(0), skipPartialLine_(false) {
        bool atLineStart = true;
        lineNo_ = linesBefore(fp, position, &atLineStart);
        skipPartialLine_ = !atLineStart;
    }

    long lineNo() const { return lineNo_; }

    // Reads the next record into the five outputs. Returns false at a clean end
    // of data -- no further non-blank, non-comment line. Anything else that is
    // not a well-formed five-field record is diagnosed with its line number and
    // the process exits nonzero.
    bool next(int &f1, int &f2, int &f3, int &f4, mpf_class &value) {
        static const char *fieldName[4] = {"", "block", "row", "column"};
        while (true) {
            bool overlong = false;
            if (!readPhysicalLine(fp_, line_, overlong)) {
                return false; // end of data
            }
            ++lineNo_;
            if (skipPartialLine_) {
                // the tail of the c-vector / y-vector line; not a record
                skipPartialLine_ = false;
                continue;
            }
            if (overlong) {
                fprintf(stderr, "%s: line %ld: record line exceeds %lu bytes\n", fileKind_, lineNo_, static_cast<unsigned long>(SDPA_MAX_RECORD_LINE));
                rError("io::read overlong record in input");
            }

            token_.clear();
            std::string current;
            for (size_t p = 0; p < line_.size(); ++p) {
                if (isFieldDelim(line_[p])) {
                    if (!current.empty()) {
                        token_.push_back(current);
                        current.clear();
                    }
                } else {
                    current.push_back(line_[p]);
                }
            }
            if (!current.empty()) {
                token_.push_back(current);
            }

            if (token_.empty()) {
                continue; // blank line
            }
            if (token_[0][0] == '*' || token_[0][0] == '"') {
                continue; // comment line, same convention as the header
            }

            if (token_.size() != 5) {
                fprintf(stderr,
                        "%s: line %ld: expected 5 fields <%s> <block> <row> <column> <value> on one line, found %lu: '%s'\n",
                        fileKind_, lineNo_, firstFieldName_, static_cast<unsigned long>(token_.size()), clipForMessage(line_).c_str());
                rError("io::read malformed record in input");
            }

            int parsed[4];
            for (int t = 0; t < 4; ++t) {
                if (!parseFieldInt(token_[t], parsed[t])) {
                    fprintf(stderr, "%s: line %ld: field %d <%s> is not an integer: '%s'\n",
                            fileKind_, lineNo_, t + 1, t == 0 ? firstFieldName_ : fieldName[t], clipForMessage(token_[t]).c_str());
                    rError("io::read malformed record in input");
                }
            }
            if (convertReal(token_[4], value) != CONVERT_OK) {
                fprintf(stderr, "%s: line %ld: field 5 <value> is not a number: '%s'\n", fileKind_, lineNo_, clipForMessage(token_[4]).c_str());
                rError("io::read malformed record in input");
            }

            f1 = parsed[0];
            f2 = parsed[1];
            f3 = parsed[2];
            f4 = parsed[3];
            return true;
        }
    }

  private:
    FILE *fp_;
    const char *fileKind_;
    const char *firstFieldName_;
    long lineNo_;
    bool skipPartialLine_;
    std::string line_;
    std::vector<std::string> token_;
};

// ---------------------------------------------------------------------------
// Header range validation.
//
// mDIM and nBLOCK were conversion-checked but not range-checked, and they drive
// allocations directly: "new int[nBlock]" with a negative nBlock terminates on
// std::bad_array_new_length (exit 134), and "2000000000 = nBLOCK" in a 100-byte
// file successfully reserved three 8 GB arrays before anything noticed.
//
// The upper bound is taken from the file itself rather than picked: the c vector
// holds m numbers and bLOCKsTRUCT holds nBlock numbers, and no number occupies
// less than one byte, so neither count can exceed the file's own length. That
// rejects a fabricated dimension without capping a legitimately large problem.
// ---------------------------------------------------------------------------
long inputFileBound(FILE *fp) {
    const long saved = ftell(fp);
    if (saved < 0 || fseek(fp, 0, SEEK_END) != 0) {
        return LONG_MAX; // not seekable: fall back to no file-derived bound
    }
    const long size = ftell(fp);
    fseek(fp, saved, SEEK_SET);
    return size < 0 ? LONG_MAX : size;
}

// A dense block of this order already needs 46340^2 = 2.1e9 entries, and one
// more would overflow the int arithmetic used for block sizes throughout.
const int SDPA_MAX_BLOCK_SIZE = 46340;

} // namespace

// 2008/02/27  kazuhide nakata
#if 0 // not use
void IO::read(FILE* fpData, int m,
	      int SDP_nBlock,int* SDP_blockStruct,
	      int SOCP_nBlock,int* SOCP_blockStruct,
	      int LP_nBlock, 
		  int nBlock, int* blockStruct, int* blockType, int* blockNumber,
	      InputData& inputData, bool isDataSparse)
{
  inputData.initialize_bVec(m);
  read(fpData,inputData.b);
  long position = ftell(fpData);
  // C,A must be accessed "twice".

  // count numbers of elements of C and A
  int* SDP_CNonZeroCount = NULL;
  SDP_CNonZeroCount = new int[SDP_nBlock];
  if (SDP_CNonZeroCount==NULL) {
    rError("Memory exhausted about blockStruct");
  }
  int* SDP_ANonZeroCount = NULL;
  SDP_ANonZeroCount = new int[m*SDP_nBlock];
  if (SDP_ANonZeroCount==NULL) {
    rError("Memory exhausted about blockStruct");
  }

  // count numbers of elements of C and A
  int* SOCP_CNonZeroCount = NULL;
  SOCP_CNonZeroCount = new int[SOCP_nBlock];
  if (SOCP_CNonZeroCount==NULL) {
    rError("Memory exhausted about blockStruct");
  }
  int* SOCP_ANonZeroCount = NULL;
  SOCP_ANonZeroCount = new int[m*SOCP_nBlock];
  if (SOCP_ANonZeroCount==NULL) {
    rError("Memory exhausted about blockStruct");
  }

  // count numbers of elements of C and A
  bool* LP_CNonZeroCount = NULL;
  LP_CNonZeroCount = new bool[LP_nBlock];
  if (LP_CNonZeroCount==NULL) {
    rError("Memory exhausted about blockStruct");
  }
  bool* LP_ANonZeroCount = NULL;
  LP_ANonZeroCount = new bool[m*LP_nBlock];
  if (LP_ANonZeroCount==NULL) {
    rError("Memory exhausted about blockStruct");
  }

  //   initialize C and A
  read(fpData,m,
       SDP_nBlock, SDP_blockStruct, SDP_CNonZeroCount, SDP_ANonZeroCount,
       SOCP_nBlock, SOCP_blockStruct, SOCP_CNonZeroCount, SOCP_ANonZeroCount,
       LP_nBlock, LP_CNonZeroCount, LP_ANonZeroCount,
	   nBlock, blockStruct, blockType, blockNumber,
       isDataSparse);
  //   rMessage(" C and A count over");
  inputData.initialize_CMat(SDP_nBlock, SDP_blockStruct,
			    SDP_CNonZeroCount,
			    SOCP_nBlock,  SOCP_blockStruct,
			    SOCP_CNonZeroCount,
			    LP_nBlock, LP_CNonZeroCount);
  inputData.initialize_AMat(m,SDP_nBlock, SDP_blockStruct,
			    SDP_ANonZeroCount,
			    SOCP_nBlock,  SOCP_blockStruct,
			    SOCP_ANonZeroCount,
			    LP_nBlock, LP_ANonZeroCount);
  delete[] SDP_CNonZeroCount;
  SDP_CNonZeroCount = NULL;
  delete[] SDP_ANonZeroCount;
  SDP_ANonZeroCount = NULL;
  delete[] SOCP_CNonZeroCount;
  SOCP_CNonZeroCount = NULL;
  delete[] SOCP_ANonZeroCount;
  SOCP_ANonZeroCount = NULL;
  delete[] LP_CNonZeroCount;
  LP_CNonZeroCount = NULL;
  delete[] LP_ANonZeroCount;
  LP_ANonZeroCount = NULL;
    
  //   rMessage(" C and A initialize over");
  read(fpData, inputData, m, 
       SDP_nBlock, SDP_blockStruct, 
       SOCP_nBlock, SOCP_blockStruct, 
       LP_nBlock, 
	   nBlock, blockStruct, blockType, blockNumber,
       position, isDataSparse);
  //   rMessage(" C and A have been read");
}
#endif

void IO::read(FILE *fpData, FILE *fpout, int &m, char *str) {
    while (true) {
        volatile int dummy = 0; // for gcc-3.3 bug
        if (fgets(str, lengthOfString, fpData) == NULL) {
            rError("IO::read:: unexpected end of file while reading the SDPA header");
        }
        if (str[0] == '*' || str[0] == '"') {
            fprintf(fpout, "%s", str);
        } else {
            if (sscanf(str, "%d", &m) <= 0) {
                reportBadLine("SDPA data file: the mDim record is not an integer", str);
                rError("IO::read:: invalid mDim record in the SDPA header");
            }
            // Range-check BEFORE m reaches initialize_bVec(m), new vector<int>[m+1]
            // and new SparseLinearSpace[m].
            if (m < 1) {
                fprintf(stderr, "SDPA data file: mDIM must be at least 1, found %d\n", m);
                rError("IO::read:: mDIM out of range in the SDPA header");
            }
            {
                const long bound = inputFileBound(fpData);
                if (static_cast<long>(m) > bound) {
                    fprintf(stderr, "SDPA data file: mDIM = %d, but the whole file is only %ld bytes and cannot carry that many c-vector entries\n", m, bound);
                    rError("IO::read:: mDIM out of range in the SDPA header");
                }
            }
            break;
        }
    }
}

void IO::read(FILE *fpData, int &nBlock) {
    if (fscanf(fpData, "%d", &nBlock) <= 0) {
        fprintf(stderr, "SDPA data file: the nBlock record is missing or is not an integer\n");
        rError("IO::read:: invalid nBlock record in the SDPA header");
    }
    // Range-check BEFORE nBlock reaches the three new int[nBlock] in main().
    if (nBlock < 1) {
        fprintf(stderr, "SDPA data file: nBLOCK must be at least 1, found %d\n", nBlock);
        rError("IO::read:: nBLOCK out of range in the SDPA header");
    }
    const long bound = inputFileBound(fpData);
    if (static_cast<long>(nBlock) > bound) {
        fprintf(stderr, "SDPA data file: nBLOCK = %d, but the whole file is only %ld bytes and cannot carry that many bLOCKsTRUCT entries\n", nBlock, bound);
        rError("IO::read:: nBLOCK out of range in the SDPA header");
    }
}

void IO::read(FILE *fpData, int nBlock, int *blockStruct) {
    // Every block size is validated as it is read, and the totals are accumulated
    // in long long, so that main()'s own loop -- which negates a negative entry
    // and sums the LP dimensions in int -- cannot be handed a value that
    // overflows. A blockStruct entry of INT_MIN would make "-blockStruct[i]"
    // undefined; a long run of large negative entries would wrap LP_nBlock.
    long long totalLP = 0;
    for (int l = 0; l < nBlock; ++l) {
        if (fscanf(fpData, "%*[^0-9+-]%d", &blockStruct[l]) <= 0) {
            fprintf(stderr, "SDPA data file: bLOCKsTRUCT entry %d of %d is missing or is not an integer\n", l + 1, nBlock);
            rError("IO::read:: invalid bLOCKsTRUCT record in the SDPA header");
        }
        if (blockStruct[l] == 0) {
            fprintf(stderr, "SDPA data file: bLOCKsTRUCT entry %d of %d is 0; a block must have a nonzero size\n", l + 1, nBlock);
            rError("IO::read:: invalid bLOCKsTRUCT record in the SDPA header");
        }
        if (blockStruct[l] > SDPA_MAX_BLOCK_SIZE || blockStruct[l] < -SDPA_MAX_BLOCK_SIZE) {
            fprintf(stderr, "SDPA data file: bLOCKsTRUCT entry %d of %d is %d, outside the supported range [-%d,%d] (excluding 0)\n", l + 1, nBlock, blockStruct[l], SDPA_MAX_BLOCK_SIZE, SDPA_MAX_BLOCK_SIZE);
            rError("IO::read:: invalid bLOCKsTRUCT record in the SDPA header");
        }
        if (blockStruct[l] < 0) {
            totalLP += -static_cast<long long>(blockStruct[l]);
        } else if (blockStruct[l] == 1) {
            totalLP += 1;
        }
        if (totalLP > INT_MAX) {
            fprintf(stderr, "SDPA data file: the total LP dimension exceeds %d by bLOCKsTRUCT entry %d of %d\n", INT_MAX, l + 1, nBlock);
            rError("IO::read:: invalid bLOCKsTRUCT record in the SDPA header");
        }
    }
}

void IO::read(FILE *fpData, Vector &b) {
    for (int k = 0; k < b.nDim; ++k) {
        requireReal(fpData, b.ele[k], "the data file's c vector", k + 1, b.nDim);
    }
}

void IO::read(FILE *fpData, DenseLinearSpace &xMat, Vector &yVec, DenseLinearSpace &zMat, int nBlock, int *blockStruct, int *blockType, int *blockNumber, bool inputSparse) {
    // read initial point
    int SDP_nBlock = xMat.SDP_nBlock;
    int SOCP_nBlock = xMat.SOCP_nBlock;
    int LP_nBlock = xMat.LP_nBlock;

    // yVec is opposite sign
    for (int k = 0; k < yVec.nDim; ++k) {
        mpf_class tmp;
        requireReal(fpData, tmp, "the initial point file's y vector", k + 1, yVec.nDim);
        yVec.ele[k] = -tmp;
        //     rMessage("yVec.ele[" << k << "] = " << tmp);
    }

    if (inputSparse) {
        // sparse case , zMat , xMat in this order
        // The block index l was previously checked only from below (l < 1) and was
        // then DISCARDED: every block past the SDP and SOCP partitions was treated
        // as LP and the element written to setElement_LP(i - 1). Two separate
        // defects came out of that.
        //
        //  * l had no upper bound, so "2 999 1 1 5.9" was accepted, the run exited
        //    0 and printed a solution.
        //  * Even a CORRECT l was ignored, so on a problem with two or more LP
        //    blocks a well-formed initial point was written to the wrong flattened
        //    slots -- block 3's local indices 1,2 overwrote block 2's slots 0,1 and
        //    block 2's own values never landed anywhere. Deleting block 2's records
        //    from a correct initial point produced byte-identical output to
        //    supplying them. That is a wrong-answer defect on VALID input, not
        //    merely a missing bound.
        //
        // The data reader already had this right (blockNumber[l-1] + i - 1); this
        // now mirrors it exactly, which also fixes the SDP branch for a problem
        // whose blocks are not ordered SDP-first -- "l <= SDP_nBlock" silently
        // mistook an SDP block for an LP one whenever an LP block came before it
        // in bLOCKsTRUCT.
        int i, j, l, target;
        mpf_class value;
        SparseRecordReader record(fpData, "SDPA initial point file", "target", ftell(fpData));
        while (record.next(target, l, i, j, value)) {
#if 0
      rMessage("target = " << target
	       << ": l " << l
	       << ": i " << i
	       << ": j " << j
	       << ": value " <<value);
#endif

            if (target != 1 && target != 2) {
                fprintf(stderr, "SDPA initial point file: line %ld: target out of range [1,2] in record (target=%d, l=%d, i=%d, j=%d)\n", record.lineNo(), target, l, i, j);
                rError("io::read invalid target in initial point file");
            }
            if (l < 1 || l > nBlock) {
                fprintf(stderr, "SDPA initial point file: line %ld: block index out of range [1,%d] in record (target=%d, l=%d, i=%d, j=%d)\n", record.lineNo(), nBlock, target, l, i, j);
                rError("io::read invalid block index in initial point file");
            }
            const int blockSize = blockStruct[l - 1];

            if (blockType[l - 1] == 1) {
                // SDP part
                if (i < 1 || i > blockSize || j < 1 || j > blockSize) {
                    fprintf(stderr, "SDPA initial point file: line %ld: entry index out of range [1,%d] in record (target=%d, l=%d, i=%d, j=%d)\n", record.lineNo(), blockSize, target, l, i, j);
                    rError("io::read invalid entry index in initial point file");
                }
                const int l2 = blockNumber[l - 1];
                if (target == 1) {
                    zMat.setElement_SDP(l2, i - 1, j - 1, value);
                } else {
                    xMat.setElement_SDP(l2, i - 1, j - 1, value);
                }
            } else if (blockType[l - 1] == 2) {
                // SOCP part
                rError("io:: current version does not support SOCP");
            } else if (blockType[l - 1] == 3) {
                // LP part
                if (i != j) {
                    fprintf(stderr, "SDPA initial point file: line %ld: an LP block entry must be diagonal, but row != column in record (target=%d, l=%d, i=%d, j=%d)\n", record.lineNo(), target, l, i, j);
                    rError("io:: LP part  3rd elemtn != 4th elemnt");
                }
                if (i < 1 || i > blockSize) {
                    fprintf(stderr, "SDPA initial point file: line %ld: entry index out of range [1,%d] in record (target=%d, l=%d, i=%d, j=%d)\n", record.lineNo(), blockSize, target, l, i, j);
                    rError("io::read invalid entry index in initial point file");
                }
                const int lp = blockNumber[l - 1] + i - 1;
                if (target == 1) {
                    zMat.setElement_LP(lp, value);
                } else {
                    xMat.setElement_LP(lp, value);
                }
            } else {
                rError("io::read not valid blockType");
            }
        } // end of 'while (record.next(...))'
    } else {
        /* MODIFIED (review2 finding 2), 2026-08-08: dense case, zMat then xMat.
           The file is written in the ORIGINAL bLOCKsTRUCT order, so a reader
           that consumes every compacted SDP block first and the flattened LP
           part afterwards mis-assigns every value once LP and SDP blocks are
           interleaved: with block structure {-1, 2} the LP scalar was consumed
           as the first SDP entry, and a mathematically valid positive definite
           initial point was rejected with "initial point is not positive
           definite" while its sparse spelling reached pdOPT. Iterate the
           original blocks and dispatch on blockType, exactly as the sparse
           branch does; blockNumber[] maps each original block to its compacted
           SDP index or flat LP offset. A dense LP (diagonal) block of original
           size s contributes s values, matching the sparse branch's diagonal
           addressing lp = blockNumber[l] + i. */
        for (int target = 1; target <= 2; ++target) {
            for (int l = 0; l < nBlock; ++l) {
                if (blockType[l] == 1) {
                    const int b = blockNumber[l];
                    int size = (target == 1 ? zMat : xMat).SDP_block[b].nRow;
                    for (int i = 0; i < size; ++i) {
                        for (int j = 0; j < size; ++j) {
                            mpf_class tmp;
                            requireReal(fpData, tmp, "the initial point file's dense section", target, l + 1, i + 1, j + 1);
                            if (i <= j && tmp != 0.0) {
                                if (target == 1) {
                                    zMat.setElement_SDP(b, i, j, tmp);
                                } else {
                                    xMat.setElement_SDP(b, i, j, tmp);
                                }
                            }
                        }
                    }
                } else if (blockType[l] == 2) {
                    rError("io:: current version does not support SOCP");
                } else if (blockType[l] == 3) {
                    const int size = std::abs(blockStruct[l]);
                    for (int i = 0; i < size; ++i) {
                        mpf_class tmp;
                        requireReal(fpData, tmp, "the initial point file's dense section", target, l + 1, i + 1, i + 1);
                        if (tmp != 0.0) {
                            if (target == 1) {
                                zMat.setElement_LP(blockNumber[l] + i, tmp);
                            } else {
                                xMat.setElement_LP(blockNumber[l] + i, tmp);
                            }
                        }
                    }
                } else {
                    rError("io::read not valid blockType");
                }
            }
        }
    } // end of 'if (inputSparse)'
}

// 2008/02/27 kazuhide nakata
// not use

// 2008/02/27 kazuhide nakata
// without LP_ANonZeroCount
#if 1
void IO::read(FILE *fpData, int m, int SDP_nBlock, int *SDP_blockStruct, int SOCP_nBlock, int *SOCP_blockStruct, int LP_nBlock, int nBlock, int *blockStruct, int *blockType, int *blockNumber, InputData &inputData, bool isDataSparse) {
    inputData.initialize_bVec(m);
    read(fpData, inputData.b);
    long position = ftell(fpData);

    // C,A must be accessed "mpf_class".

    //   initialize block struct of C and A
    setBlockStruct(fpData, inputData, m, SDP_nBlock, SDP_blockStruct, SOCP_nBlock, SOCP_blockStruct, LP_nBlock, nBlock, blockStruct, blockType, blockNumber, position, isDataSparse);
    //   rMessage(" C and A initialize over");

    setElement(fpData, inputData, m, SDP_nBlock, SDP_blockStruct, SOCP_nBlock, SOCP_blockStruct, LP_nBlock, nBlock, blockStruct, blockType, blockNumber, position, isDataSparse);
    //   rMessage(" C and A have been read");
}
#endif

// 2008/02/27 kazuhide nakata
// without LP_ANonZeroCount
void IO::setBlockStruct(FILE *fpData, InputData &inputData, int m, int SDP_nBlock, int *SDP_blockStruct, int SOCP_nBlock, int *SOCP_blockStruct, int LP_nBlock, int nBlock, int *blockStruct, int *blockType, int *blockNumber, long position, bool isDataSparse) {
    // seed the positon of C in the fpData
    fseek(fpData, position, 0);

    vector<int> *SDP_index = NULL;
    SDP_index = new vector<int>[m + 1];
    vector<int> *SOCP_index = NULL;
    SOCP_index = new vector<int>[m + 1];
    vector<int> *LP_index = NULL;
    LP_index = new vector<int>[m + 1];

    // for SDP
    int SDP_sp_nBlock;
    int *SDP_sp_index = NULL;
    int *SDP_sp_blockStruct = NULL;
    int *SDP_sp_NonZeroNumber = NULL;
    SDP_sp_index = new int[SDP_nBlock];
    SDP_sp_blockStruct = new int[SDP_nBlock];
    SDP_sp_NonZeroNumber = new int[SDP_nBlock];
    // for SOCP
    int SOCP_sp_nBlock = 0;
    int *SOCP_sp_blockStruct = NULL;
    int *SOCP_sp_index = NULL;
    int *SOCP_sp_NonZeroNumber = NULL;
    // for LP
    int LP_sp_nBlock;
    int *LP_sp_index;
    LP_sp_index = new int[LP_nBlock];

    if (isDataSparse) {
        int i, j, k, l;
        mpf_class value;
        SparseRecordReader record(fpData, "SDPA sparse data file", "matrix", position);
        while (record.next(k, l, i, j, value)) {

            if (k < 0 || k > m) {
                fprintf(stderr, "SDPA sparse data file: line %ld: matrix index out of range [0,%d] in record (k=%d, l=%d, i=%d, j=%d)\n", record.lineNo(), m, k, l, i, j);
                rError("io::read invalid matrix index in input data");
            }
            if (l < 1 || l > nBlock) {
                fprintf(stderr, "SDPA sparse data file: line %ld: block index out of range [1,%d] in record (k=%d, l=%d, i=%d, j=%d)\n", record.lineNo(), nBlock, k, l, i, j);
                rError("io::read invalid block index in input data");
            }
            if (i < 1 || i > blockStruct[l - 1] || j < 1 || j > blockStruct[l - 1]) {
                fprintf(stderr, "SDPA sparse data file: line %ld: entry index out of range [1,%d] in record (k=%d, l=%d, i=%d, j=%d)\n", record.lineNo(), blockStruct[l - 1], k, l, i, j);
                rError("io::read invalid entry index in input data");
            }

            if (blockType[l - 1] == 1) { // SDP part
                int l2 = blockNumber[l - 1];
                SDP_index[k].push_back(l2);
            } else if (blockType[l - 1] == 2) { // SOCP part
                rError("io:: current version does not support SOCP");
                int l2 = blockNumber[l - 1];
                ;
                SOCP_index[k].push_back(l2);
            } else if (blockType[l - 1] == 3) { // LP part
                if (i != j) {
                    fprintf(stderr, "SDPA sparse data file: line %ld: an LP block entry must be diagonal, but row != column in record (k=%d, l=%d, i=%d, j=%d)\n", record.lineNo(), k, l, i, j);
                    rError("IO::initializeLinearSpace");
                }
                int l2 = blockNumber[l - 1];
                LP_index[k].push_back(l2 + i - 1);
            } else {
                rError("io::read not valid blockType");
            }
        } // end of 'while (true)'

    } else { // isDataSparse == false

        // k==0
        for (int l2 = 0; l2 < nBlock; ++l2) {
            if (blockType[l2] == 1) { // SDP part
                int l = blockNumber[l2];
                int size = SDP_blockStruct[l];
                for (int i = 0; i < size; ++i) {
                    for (int j = 0; j < size; ++j) {
                        mpf_class tmp;
                        requireReal(fpData, tmp, "the dense data file's C matrix (matrix, block, row, column)", 0, l2 + 1, i + 1, j + 1);
                        if (i <= j && tmp != 0.0) {
                            SDP_index[0].push_back(l);
                        }
                    }
                }
            } else if (blockType[l2] == 2) { // SOCP part
                rError("io:: current version does not support SOCP");
            } else if (blockType[l2] == 3) { // LP part
                for (int j = 0; j < blockStruct[l2]; ++j) {
                    mpf_class tmp;
                    requireReal(fpData, tmp, "the dense data file's C matrix (matrix, block, row, column)", 0, l2 + 1, j + 1, j + 1);
                    if (tmp != 0.0) {
                        LP_index[0].push_back(blockNumber[l2] + j);
                    }
                }
            } else {
                rError("io::read not valid blockType");
            }
        }

        for (int k = 0; k < m; ++k) {
            // k>0
            for (int l2 = 0; l2 < nBlock; ++l2) {
                if (blockType[l2] == 1) { // SDP part
                    int l = blockNumber[l2];
                    int size = SDP_blockStruct[l];
                    for (int i = 0; i < size; ++i) {
                        for (int j = 0; j < size; ++j) {
                            mpf_class tmp;
                            requireReal(fpData, tmp, "the dense data file's A matrix (matrix, block, row, column)", k + 1, l2 + 1, i + 1, j + 1);
                            if (i <= j && tmp != 0.0) {
                                SDP_index[k + 1].push_back(l);
                            }
                        }
                    }
                } else if (blockType[l2] == 2) { // SOCP part
                    rError("io:: current version does not support SOCP");
                } else if (blockType[l2] == 3) { // LP part
                    for (int j = 0; j < blockStruct[l2]; ++j) {
                        mpf_class tmp;
                        requireReal(fpData, tmp, "the dense data file's A matrix (matrix, block, row, column)", k + 1, l2 + 1, j + 1, j + 1);
                        if (tmp != 0.0) {
                            LP_index[k + 1].push_back(blockNumber[l2] + j);
                        }
                    }
                } else {
                    rError("io::read not valid blockType");
                }
            }
        }

    } // end of 'if (isDataSparse)'

    inputData.A = new SparseLinearSpace[m];
    for (int k = 0; k < m + 1; k++) {
        sort(SDP_index[k].begin(), SDP_index[k].end());
        SDP_sp_nBlock = 0;
        int previous_index = -1;
        int index;
        for (int i = 0; i < SDP_index[k].size(); i++) {
            index = SDP_index[k][i];
            if (previous_index != index) {
                SDP_sp_index[SDP_sp_nBlock] = index;
                SDP_sp_blockStruct[SDP_sp_nBlock] = SDP_blockStruct[index];
                SDP_sp_NonZeroNumber[SDP_sp_nBlock] = 1;
                previous_index = index;
                SDP_sp_nBlock++;
            } else {
                SDP_sp_NonZeroNumber[SDP_sp_nBlock - 1]++;
            }
        }
        sort(LP_index[k].begin(), LP_index[k].end());
        LP_sp_nBlock = 0;
        previous_index = -1;
        for (int i = 0; i < LP_index[k].size(); i++) {
            index = LP_index[k][i];
            if (previous_index != index) {
                LP_sp_index[LP_sp_nBlock] = index;
                previous_index = index;
                LP_sp_nBlock++;
            }
        }

        if (k == 0) {
            inputData.C.initialize(SDP_sp_nBlock, SDP_sp_index, SDP_sp_blockStruct, SDP_sp_NonZeroNumber, SOCP_sp_nBlock, SOCP_sp_blockStruct, SOCP_sp_index, SOCP_sp_NonZeroNumber, LP_sp_nBlock, LP_sp_index);
        } else {
            inputData.A[k - 1].initialize(SDP_sp_nBlock, SDP_sp_index, SDP_sp_blockStruct, SDP_sp_NonZeroNumber, SOCP_sp_nBlock, SOCP_sp_blockStruct, SOCP_sp_index, SOCP_sp_NonZeroNumber, LP_sp_nBlock, LP_sp_index);
        }
    }

    delete[] SDP_index;
    SDP_index = NULL;
    delete[] SOCP_index;
    SOCP_index = NULL;
    delete[] LP_index;
    LP_index = NULL;

    delete[] SDP_sp_index;
    SDP_sp_index = NULL;
    delete[] SDP_sp_blockStruct;
    SDP_sp_blockStruct = NULL;
    delete[] SDP_sp_NonZeroNumber;
    SDP_sp_NonZeroNumber = NULL;
#if 0
  delete[] SOCP_sp_index;
  SOCP_sp_index = NULL;
  delete[] SOCP_sp_blockStruct;
  SOCP_sp_blockStruct = NULL;
  delete[] SOCP_sp_NonZeroNumber;
  SOCP_sp_NonZeroNumber = NULL;
#endif
    delete[] LP_sp_index;
    LP_sp_index = NULL;
}

// 2008/02/27 kazuhide nakata
// without LP_ANonZeroCount
void IO::setElement(FILE *fpData, InputData &inputData, int m, int SDP_nBlock, int *SDP_blockStruct, int SOCP_nBlock, int *SOCP_blockStruct, int LP_nBlock, int nBlock, int *blockStruct, int *blockType, int *blockNumber, long position, bool isDataSparse) {
    // in Sparse, read C,A[k]

    // seed the positon of C in the fpData
    fseek(fpData, position, 0);

    if (isDataSparse) {
        int i, j, k, l;
        mpf_class value;
        SparseRecordReader record(fpData, "SDPA sparse data file", "matrix", position);
        while (record.next(k, l, i, j, value)) {
#if 0
      rMessage("input k:" << k <<
	       " l:" << l <<
	       " i:" << i <<
	       " j:" << j);
#endif

            if (k < 0 || k > m) {
                fprintf(stderr, "SDPA sparse data file: line %ld: matrix index out of range [0,%d] in record (k=%d, l=%d, i=%d, j=%d)\n", record.lineNo(), m, k, l, i, j);
                rError("io::read invalid matrix index in input data");
            }
            if (l < 1 || l > nBlock) {
                fprintf(stderr, "SDPA sparse data file: line %ld: block index out of range [1,%d] in record (k=%d, l=%d, i=%d, j=%d)\n", record.lineNo(), nBlock, k, l, i, j);
                rError("io::read invalid block index in input data");
            }
            if (i < 1 || i > blockStruct[l - 1] || j < 1 || j > blockStruct[l - 1]) {
                fprintf(stderr, "SDPA sparse data file: line %ld: entry index out of range [1,%d] in record (k=%d, l=%d, i=%d, j=%d)\n", record.lineNo(), blockStruct[l - 1], k, l, i, j);
                rError("io::read invalid entry index in input data");
            }

            if (blockType[l - 1] == 1) { // SDP part
                int l2 = blockNumber[l - 1];
                if (k == 0) {
                    inputData.C.setElement_SDP(l2, i - 1, j - 1, -value);
                } else {
                    inputData.A[k - 1].setElement_SDP(l2, i - 1, j - 1, value);
                }
            } else if (blockType[l - 1] == 2) { // SOCP part
                rError("io:: current version does not support SOCP");
                int l2 = blockNumber[l - 1];
                if (k == 0) {
                    inputData.C.setElement_SOCP(l2, i - 1, j - 1, -value);
                } else {
                    inputData.A[k - 1].setElement_SOCP(l2, i - 1, j - 1, value);
                }
            } else if (blockType[l - 1] == 3) { // LP part
                if (i != j) {
                    fprintf(stderr, "SDPA sparse data file: line %ld: an LP block entry must be diagonal, but row != column in record (k=%d, l=%d, i=%d, j=%d)\n", record.lineNo(), k, l, i, j);
                    rError("io:: LP part  3rd elemtn != 4th elemnt");
                }
                if (k == 0) {
                    inputData.C.setElement_LP(blockNumber[l - 1] + i - 1, -value);
                } else {
                    inputData.A[k - 1].setElement_LP(blockNumber[l - 1] + i - 1, value);
                }
            } else {
                rError("io::read not valid blockType");
            }
        }
    } else { // dense

        // k==0
        for (int l2 = 0; l2 < nBlock; ++l2) {
            if (blockType[l2] == 1) { // SDP part
                int l = blockNumber[l2];
                int size = SDP_blockStruct[l];
                for (int i = 0; i < size; ++i) {
                    for (int j = 0; j < size; ++j) {
                        mpf_class tmp;
                        requireReal(fpData, tmp, "the dense data file's C matrix (matrix, block, row, column)", 0, l2 + 1, i + 1, j + 1);
                        if (i <= j && tmp != 0.0) {
                            inputData.C.setElement_SDP(l, i, j, -tmp);
                        }
                    }
                }
            } else if (blockType[l2] == 2) { // SOCP part
                rError("io:: current version does not support SOCP");
            } else if (blockType[l2] == 3) { // LP part
                for (int j = 0; j < blockStruct[l2]; ++j) {
                    mpf_class tmp;
                    requireReal(fpData, tmp, "the dense data file's C matrix (matrix, block, row, column)", 0, l2 + 1, j + 1, j + 1);
                    if (tmp != 0.0) {
                        inputData.C.setElement_LP(blockNumber[l2] + j, -tmp);
                    }
                }
            } else {
                rError("io::read not valid blockType");
            }
        }

        // k > 0
        for (int k = 0; k < m; ++k) {

            for (int l2 = 0; l2 < nBlock; ++l2) {
                if (blockType[l2] == 1) { // SDP part
                    int l = blockNumber[l2];
                    int size = SDP_blockStruct[l];
                    for (int i = 0; i < size; ++i) {
                        for (int j = 0; j < size; ++j) {
                            mpf_class tmp;
                            requireReal(fpData, tmp, "the dense data file's A matrix (matrix, block, row, column)", k + 1, l2 + 1, i + 1, j + 1);
                            if (i <= j && tmp != 0.0) {
                                inputData.A[k].setElement_SDP(l, i, j, tmp);
                            }
                        }
                    }
                } else if (blockType[l2] == 2) { // SOCP part
                    rError("io:: current version does not support SOCP");
                } else if (blockType[l2] == 3) { // LP part
                    for (int j = 0; j < blockStruct[l2]; ++j) {
                        mpf_class tmp;
                        requireReal(fpData, tmp, "the dense data file's A matrix (matrix, block, row, column)", k + 1, l2 + 1, j + 1, j + 1);
                        if (tmp != 0.0) {
                            inputData.A[k].setElement_LP(blockNumber[l2] + j, tmp);
                        }
                    }
                } else {
                    rError("io::read not valid blockType");
                }
            }

        } // for k

    } // end of 'if (isDataSparse)'
}

void IO::printHeader(FILE *fpout, FILE *Display) {
    if (fpout) {
        fprintf(fpout, "   mu      thetaP  thetaD  objP      objD "
                       "     alphaP  alphaD  beta \n");
    }
    if (Display) {
        fprintf(Display, "   mu      thetaP  thetaD  objP      objD "
                         "     alphaP  alphaD  beta \n");
    }
}

void IO::printOneIteration(int pIteration, AverageComplementarity &mu, RatioInitResCurrentRes &theta, SolveInfo &solveInfo, StepLength &alpha, DirectionParameter &beta, FILE *fpout, FILE *Display) {
#if REVERSE_PRIMAL_DUAL
    if (Display) {
        mpf_class mtmp1 = -solveInfo.objValDual;
        mpf_class mtmp2 = -solveInfo.objValPrimal;
        gmp_fprintf(Display,
                    "%2d %4.1Fe %4.1Fe %4.1Fe %+7.2Fe %+7.2Fe"
                    " %4.1Fe %4.1Fe %4.F2e\n",
                    pIteration, mu.current.get_mpf_t(), theta.dual.get_mpf_t(), theta.primal.get_mpf_t(), mtmp1.get_mpf_t(), mtmp2.get_mpf_t(), alpha.dual.get_mpf_t(), alpha.primal.get_mpf_t(), beta.value.get_mpf_t());
    }
    if (fpout) {
        mpf_class mtmp1 = -solveInfo.objValDual;
        mpf_class mtmp2 = -solveInfo.objValPrimal;
        gmp_fprintf(fpout,
                    "%2d %4.1Fe %4.1Fe %4.1Fe %+7.2Fe %+7.2Fe"
                    " %4.1Fe %4.1Fe %4.2Fe\n",
                    pIteration, mu.current.get_mpf_t(), theta.dual.get_mpf_t(), theta.primal.get_mpf_t(), mtmp1.get_mpf_t(), mtmp2.get_mpf_t(), alpha.dual.get_mpf_t(), alpha.primal.get_mpf_t(), beta.value.get_mpf_t());
    }
#else
    if (Display) {
        gmp_fprintf(Display,
                    "%2d %4.1Fe %4.1Fe %4.1Fe %+7.2Fe %+7.2Fe"
                    " %4.1Fe %4.1Fe %4.2Fe\n",
                    pIteration.get_mpf_t(), mu.current.get_mpf_t(), theta.primal.get_mpf_t(), theta.dual.get_mpf_t(), solveInfo.objValPrimal.get_mpf_t(), solveInfo.objValDual.get_mpf_t(), alpha.primal.get_mpf_t(), alpha.dual.get_mpf_t(), beta.value.get_mpf_t());
    }
    if (fpout) {
        gmp_fprintf(fpout,
                    "%2d %4.1Fe %4.1Fe %4.1Fe %+7.2Fe %+7.2Fe"
                    " %4.1Fe %4.1Fe %4.2Fe\n",
                    pIteration.get_mpf_t(), mu.current.get_mpf_t(), theta.primal.get_mpf_t(), theta.dual.get_mpf_t(), solveInfo.objValPrimal.get_mpf_t(), solveInfo.objValDual.get_mpf_t(), alpha.primal.get_mpf_t(), alpha.dual.get_mpf_t(), beta.value.get_mpf_t());
    }
#endif
}

mpf_class fast_approx_log10(const mpf_class &value) {
    // Get the mantissa (a) and exponent (b) such that value = a * 2^b
    mp_exp_t exponent;
    double mantissa = mpf_get_d_2exp(&exponent, value.get_mpf_t());

    // Calculate log10(a) + b * log10(2)
    double log10_mantissa = std::log10(mantissa);
    double log10_2 = std::log10(2.0);
    return log10_mantissa + exponent * log10_2;
}

void IO::printLastInfo(int pIteration, AverageComplementarity &mu, RatioInitResCurrentRes &theta, SolveInfo &solveInfo, StepLength &alpha, DirectionParameter &beta, Residuals &currentRes, Phase &phase, Solutions &currentPt, double cputime, InputData &inputData, WorkVariables &work, ComputeTime &com, Parameter &param, FILE *fpout, FILE *Display, bool printTime) {
    int nDim = currentPt.nDim;

    printOneIteration(pIteration, mu, theta, solveInfo, alpha, beta, fpout, Display);

    mpf_class mean = (abs(solveInfo.objValPrimal) + abs(solveInfo.objValDual)) / 2.0;
    mpf_class PDgap = abs(solveInfo.objValPrimal - solveInfo.objValDual);
    // mpf_class dominator;
    mpf_class relgap;
    if (mean < 1.0) {
        relgap = PDgap;
    } else {
        relgap = PDgap / mean;
    }

    mpf_class gap = mu.current * nDim;
    mpf_class digits = -fast_approx_log10(abs(PDgap / mean));

#if DIMACS_PRINT
    mpf_class tmp = 0.0;
    mpf_class b1 = 0.0;
    for (int k = 0; k < inputData.b.nDim; ++k) {
        tmp = abs(inputData.b.ele[k]);
        b1 = max(b1, tmp);
    }
    mpf_class c1 = 0.0;
    for (int l = 0; l < inputData.C.SDP_sp_nBlock; ++l) {
        SparseMatrix &Cl = inputData.C.SDP_sp_block[l];
        if (Cl.type == SparseMatrix::SPARSE) {
            for (int i = 0; i < Cl.NonZeroCount; ++i) {
                tmp = abs(Cl.sp_ele[i]);
                c1 = max(c1, tmp);
            }
        } else if (Cl.type == SparseMatrix::DENSE) {
            for (int i = 0; i < Cl.nRow * Cl.nCol; ++i) {
                tmp = abs(Cl.de_ele[i]);
                c1 = max(c1, tmp);
            }
        }
    }
    for (int l = 0; l < inputData.C.SOCP_sp_nBlock; ++l) {
        rError("io:: current version does not support SOCP");
    }
    for (int l = 0; l < inputData.C.LP_sp_nBlock; ++l) {
        tmp = abs(inputData.C.LP_sp_block[l]);
        c1 = max(c1, tmp);
    }
    mpf_class p_norm;
    Lal::let(tmp, '=', currentRes.primalVec, '.', currentRes.primalVec);
    p_norm = sqrt(tmp);
    mpf_class d_norm = 0.0;
    for (int l = 0; l < currentRes.dualMat.SDP_nBlock; ++l) {
        Lal::let(tmp, '=', currentRes.dualMat.SDP_block[l], '.', currentRes.dualMat.SDP_block[l]);
        d_norm += sqrt(tmp);
    }
    for (int l = 0; l < currentRes.dualMat.SOCP_nBlock; ++l) {
        rError("io:: current version does not support SOCP");
    }
    tmp = 0.0;
    for (int l = 0; l < currentRes.dualMat.LP_nBlock; ++l) {
        tmp += currentRes.dualMat.LP_block[l] * currentRes.dualMat.LP_block[l];
    }
    d_norm += sqrt(tmp);
    mpf_class x_min = Jal::getMinEigen(currentPt.xMat, work);
    mpf_class z_min = Jal::getMinEigen(currentPt.zMat, work);

    // gmp_printf("b1:%Fe\n",b1);
    // gmp_printf("c1:%Fe\n",c1);
    // gmp_printf("p_norm:%Fe\n",p_norm);
    // gmp_printf("d_norm:%Fe\n",d_norm);
    // gmp_printf("x_min:%Fe\n",x_min);
    // gmp_printf("z_min:%Fe\n",z_min);

    mpf_class ctx = solveInfo.objValPrimal;
    mpf_class bty = solveInfo.objValDual;
    mpf_class xtz = 0.0;
    Lal::let(xtz, '=', currentPt.xMat, '.', currentPt.zMat);

    mpf_class mzero = 0.0;
    mpf_class err1 = p_norm / (1 + b1);
    mpf_class err2 = max(mzero, -x_min / (1 + b1));
    mpf_class err3 = d_norm / (1 + c1);
    mpf_class err4 = max(mzero, -z_min / (1 + c1));
    mpf_class err5 = (ctx - bty) / (1 + abs(ctx) + abs(bty));
    mpf_class err6 = xtz / (1 + abs(ctx) + abs(bty));

#endif
    if (Display) {
        fprintf(Display, "\n");
        phase.display(Display);
        fprintf(Display, "   Iteration = %d\n", pIteration);
        gmp_fprintf(Display, "          mu = %4.16Fe\n", mu.current.get_mpf_t());
        gmp_fprintf(Display, "relative gap = %4.16Fe\n", relgap.get_mpf_t());
        gmp_fprintf(Display, "         gap = %4.16Fe\n", gap.get_mpf_t());
        gmp_fprintf(Display, "      digits = %4.16Fe\n", digits.get_mpf_t());

#if REVERSE_PRIMAL_DUAL
        mpf_class mtmp1 = -solveInfo.objValDual;
        mpf_class mtmp2 = -solveInfo.objValPrimal;
        gmp_fprintf(Display, "objValPrimal = %10.16Fe\n", mtmp1.get_mpf_t());
        gmp_fprintf(Display, "objValDual   = %10.16Fe\n", mtmp2.get_mpf_t());
        gmp_fprintf(Display, "p.feas.error = %10.16Fe\n", currentRes.normDualMat.get_mpf_t());
        gmp_fprintf(Display, "d.feas.error = %10.16Fe\n", currentRes.normPrimalVec.get_mpf_t());
        gmp_fprintf(Display, "relative eps = %10.16Fe\n", Rlamch_gmp("E").get_mpf_t());
#else
        gmp_fprintf(Display, "objValPrimal = %10.16Fe\n", solveInfo.objValPrimal.get_mpf_t());
        gmp_fprintf(Display, "objValDual   = %10.16Fe\n", solveInfo.objValDual.get_mpf_t());
        gmp_fprintf(Display, "p.feas.error = %10.16Fe\n", currentRes.normPrimalVec.get_mpf_t());
        gmp_fprintf(Display, "d.feas.error = %10.16Fe\n", currentRes.normDualMat.get_mpf_t());
        gmp_fprintf(Display, "relative eps = %10.16Fe\n", gmp_dlamchE().get_mpf_t());
#endif
        if (printTime == true) {
            fprintf(Display, "total time   = %.3f\n", cputime);
        }
#if DIMACS_PRINT
        fprintf(Display, "\n");
        gmp_fprintf(Display, "* DIMACS_ERRORS * \n");
        gmp_fprintf(Display, "err1 = %4.16Fe  [%40s]\n", err1.get_mpf_t(), "||Ax-b|| / (1+||b||_1) ");
        gmp_fprintf(Display, "err2 = %4.16Fe  [%40s]\n", err2.get_mpf_t(), "max(0, -lambda(x) / (1+||b||_1))");
        gmp_fprintf(Display, "err3 = %4.16Fe  [%40s]\n", err3.get_mpf_t(), "||A^Ty + z - c || / (1+||c||_1) ");
        gmp_fprintf(Display, "err4 = %4.16Fe  [%40s]\n", err4.get_mpf_t(), "max(0, -lambda(z) / (1+||c||_1))");
        gmp_fprintf(Display, "err5 = %4.16Fe  [%40s]\n", err5.get_mpf_t(), "(<c,x> - by) / (1 + |<c,x>| + |by|)");
        gmp_fprintf(Display, "err6 = %4.16Fe  [%40s]\n", err6.get_mpf_t(), "<x,z> / (1 + |<c,x>| + |by|)");
        fprintf(Display, "\n");
#endif
    }
    if (fpout) {
        fprintf(fpout, "\n");
        phase.display(fpout);
        fprintf(fpout, "   Iteration = %d\n", pIteration);
        gmp_fprintf(fpout, "          mu = %4.16Fe\n", mu.current.get_mpf_t());
        gmp_fprintf(fpout, "relative gap = %4.16Fe\n", relgap.get_mpf_t());
        gmp_fprintf(fpout, "         gap = %4.16Fe\n", gap.get_mpf_t());
        gmp_fprintf(fpout, "      digits = %4.16Fe\n", digits.get_mpf_t());

#if REVERSE_PRIMAL_DUAL
        mpf_class mtmp1 = -solveInfo.objValDual;
        mpf_class mtmp2 = -solveInfo.objValPrimal;
        gmp_fprintf(fpout, "objValPrimal = %10.16Fe\n", mtmp1.get_mpf_t());
        gmp_fprintf(fpout, "objValDual   = %10.16Fe\n", mtmp2.get_mpf_t());
        gmp_fprintf(fpout, "p.feas.error = %10.16Fe\n", currentRes.normDualMat.get_mpf_t());
        gmp_fprintf(fpout, "d.feas.error = %10.16Fe\n", currentRes.normPrimalVec.get_mpf_t());
        gmp_fprintf(fpout, "relative eps = %10.16Fe\n", Rlamch_gmp("E").get_mpf_t());
#else
        gmp_fprintf(fpout, "objValPrimal = %10.16Fe\n", solveInfo.objValPrimal.get_mpf_t());
        gmp_fprintf(fpout, "objValDual   = %10.16Fe\n", solveInfo.objValDual.get_mpf_t());
        gmp_fprintf(fpout, "p.feas.error = %10.16Fe\n", currentRes.normPrimalVec.get_mpf_t());
        gmp_fprintf(fpout, "d.feas.error = %10.16Fe\n", currentRes.normDualMat.get_mpf_t());
        gmp_fprintf(fpout, "relative eps = %10.16Fe\n", Rlamch_gmp("E").get_mpf_t());
#endif
        fprintf(fpout, "total time   = %.3f\n", cputime);
#if DIMACS_PRINT
        fprintf(fpout, "\n");
        gmp_fprintf(fpout, "* DIMACS_ERRORS * \n");
        gmp_fprintf(fpout, "err1 = %4.16Fe  [%40s]\n", err1.get_mpf_t(), "||Ax-b|| / (1+||b||_1) ");
        gmp_fprintf(fpout, "err2 = %4.16Fe  [%40s]\n", err2.get_mpf_t(), "max(0, -lambda(x) / (1+||b||_1))");
        gmp_fprintf(fpout, "err3 = %4.16Fe  [%40s]\n", err3.get_mpf_t(), "||A^Ty + z - c || / (1+||c||_1) ");
        gmp_fprintf(fpout, "err4 = %4.16Fe  [%40s]\n", err4.get_mpf_t(), "max(0, -lambda(z) / (1+||c||_1))");
        gmp_fprintf(fpout, "err5 = %4.16Fe  [%40s]\n", err5.get_mpf_t(), "(<c,x> - by) / (1 + |<c,x>| + |by|)");
        gmp_fprintf(fpout, "err6 = %4.16Fe  [%40s]\n", err6.get_mpf_t(), "<x,z> / (1 + |<c,x>| + |by|)");
        fprintf(fpout, "\n");
#endif

        gmp_fprintf(fpout, "\n\nParameters are\n");
        param.display(fpout);
        com.display(fpout);

#if 1
#if REVERSE_PRIMAL_DUAL
        fprintf(fpout, "xVec = \n");
        currentPt.yVec.display(fpout, -1.0, param.xPrint);
        fprintf(fpout, "xMat = \n");
        currentPt.zMat.display(fpout, param.XPrint);
        fprintf(fpout, "yMat = \n");
        currentPt.xMat.display(fpout, param.YPrint);
#else
        fprintf(fpout, "xMat = \n");
        currentPt.xMat.displaySolution(bs, fpout, param.XPrint);
        fprintf(fpout, "yVec = \n");
        currentPt.yVec.display(fpout, 1.0, param.xPrint);
        fprintf(fpout, "zMat = \n");
        currentPt.zMat.displaySolution(bs, fpout, param.YPrint);
#endif
#endif
    }
}

void IO::printLastInfo(int pIteration, AverageComplementarity &mu, RatioInitResCurrentRes &theta, SolveInfo &solveInfo, StepLength &alpha, DirectionParameter &beta, Residuals &currentRes, Phase &phase, Solutions &currentPt, double cputime, int nBlock, int *blockStruct, int *blockType, int *blockNumber, InputData &inputData, WorkVariables &work, ComputeTime &com, Parameter &param, FILE *fpout, FILE *Display, bool printTime) {
    int nDim = currentPt.nDim;

    printOneIteration(pIteration, mu, theta, solveInfo, alpha, beta, fpout, Display);

    mpf_class mean = (abs(solveInfo.objValPrimal) + abs(solveInfo.objValDual)) / 2.0;
    mpf_class PDgap = abs(solveInfo.objValPrimal - solveInfo.objValDual);
    // mpf_class dominator;
    mpf_class relgap;
    if (mean < 1.0) {
        relgap = PDgap;
    } else {
        relgap = PDgap / mean;
    }
    mpf_class gap = mu.current * nDim;
    mpf_class digits = 0.0;
    if (PDgap != 0.0 && mean != 0.0) {
        digits = -fast_approx_log10(abs(PDgap / mean));
    }

#if DIMACS_PRINT
    mpf_class tmp = 0.0;
    mpf_class b1 = 0.0;
    for (int k = 0; k < inputData.b.nDim; ++k) {
        tmp = abs(inputData.b.ele[k]);
        b1 = max(b1, tmp);
    }
    mpf_class c1 = 0.0;
    for (int l = 0; l < inputData.C.SDP_sp_nBlock; ++l) {
        SparseMatrix &Cl = inputData.C.SDP_sp_block[l];
        if (Cl.type == SparseMatrix::SPARSE) {
            for (int i = 0; i < Cl.NonZeroCount; ++i) {
                tmp = abs(Cl.sp_ele[i]);
                c1 = max(c1, tmp);
            }
        } else if (Cl.type == SparseMatrix::DENSE) {
            for (int i = 0; i < Cl.nRow * Cl.nCol; ++i) {
                tmp = abs(Cl.de_ele[i]);
                c1 = max(c1, tmp);
            }
        }
    }
    for (int l = 0; l < inputData.C.SOCP_sp_nBlock; ++l) {
        rError("io:: current version does not support SOCP");
    }
    for (int l = 0; l < inputData.C.LP_sp_nBlock; ++l) {
        tmp = abs(inputData.C.LP_sp_block[l]);
        c1 = max(c1, tmp);
    }
    mpf_class p_norm;
    Lal::let(tmp, '=', currentRes.primalVec, '.', currentRes.primalVec);
    p_norm = sqrt(tmp);
    mpf_class d_norm = 0.0;
    for (int l = 0; l < currentRes.dualMat.SDP_nBlock; ++l) {
        Lal::let(tmp, '=', currentRes.dualMat.SDP_block[l], '.', currentRes.dualMat.SDP_block[l]);
        d_norm += sqrt(tmp);
    }
    for (int l = 0; l < currentRes.dualMat.SOCP_nBlock; ++l) {
        rError("io:: current version does not support SOCP");
    }
    tmp = 0.0;
    for (int l = 0; l < currentRes.dualMat.LP_nBlock; ++l) {
        tmp += currentRes.dualMat.LP_block[l] * currentRes.dualMat.LP_block[l];
    }
    d_norm += sqrt(tmp);
    mpf_class x_min = Jal::getMinEigen(currentPt.xMat, work);
    mpf_class z_min = Jal::getMinEigen(currentPt.zMat, work);

    //  gmp_printf("b1:%Fe\n",b1);
    //  gmp_printf("c1:%Fe\n",c1);
    //  gmp_printf("p_norm:%Fe\n",p_norm);
    //  gmp_printf("d_norm:%Fe\n",d_norm);
    //  gmp_printf("x_min:%Fe\n",x_min);
    //  gmp_printf("z_min:%Fe\n",z_min);

    mpf_class ctx = solveInfo.objValPrimal;
    mpf_class bty = solveInfo.objValDual;
    mpf_class xtz = 0.0;
    Lal::let(xtz, '=', currentPt.xMat, '.', currentPt.zMat);

    mpf_class mzero = 0.0;
    mpf_class err1 = p_norm / (1 + b1);
    mpf_class err2 = max(mzero, -x_min / (1 + b1));
    mpf_class err3 = d_norm / (1 + c1);
    mpf_class err4 = max(mzero, -z_min / (1 + c1));
    mpf_class err5 = (ctx - bty) / (1 + abs(ctx) + abs(bty));
    mpf_class err6 = xtz / (1 + abs(ctx) + abs(bty));

#endif

    if (Display) {
        fprintf(Display, "\n");
        phase.display(Display);
        gmp_fprintf(Display, "   Iteration = %d\n", pIteration);
        gmp_fprintf(Display, "          mu = %4.16Fe\n", mu.current.get_mpf_t());
        gmp_fprintf(Display, "relative gap = %4.16Fe\n", relgap.get_mpf_t());
        gmp_fprintf(Display, "         gap = %4.16Fe\n", gap.get_mpf_t());
        gmp_fprintf(Display, "      digits = %4.16Fe\n", digits.get_mpf_t());

#if REVERSE_PRIMAL_DUAL
        mpf_class mtmp1 = -solveInfo.objValDual;
        mpf_class mtmp2 = -solveInfo.objValPrimal;
        gmp_fprintf(Display, "objValPrimal = %10.16Fe\n", mtmp1.get_mpf_t());
        gmp_fprintf(Display, "objValDual   = %10.16Fe\n", mtmp2.get_mpf_t());
        gmp_fprintf(Display, "p.feas.error = %10.16Fe\n", currentRes.normDualMat.get_mpf_t());
        gmp_fprintf(Display, "d.feas.error = %10.16Fe\n", currentRes.normPrimalVec.get_mpf_t());
        gmp_fprintf(Display, "relative eps = %10.16Fe\n", Rlamch_gmp("E").get_mpf_t());
#else
        gmp_fprintf(Display, "objValPrimal = %10.16Fe\n", solveInfo.objValPrimal.get_mpf_t());
        gmp_fprintf(Display, "objValDual   = %10.16Fe\n", solveInfo.objValDual.get_mpf_t());
        gmp_fprintf(Display, "p.feas.error = %10.16Fe\n", currentRes.normPrimalVec.get_mpf_t());
        gmp_fprintf(Display, "d.feas.error = %10.16Fe\n", currentRes.normDualMat.get_mpf_t());
        gmp_fprintf(Display, "relative eps = %10.16Fe\n", Rlamch_gmp("E").get_mpf_t());
#endif
        if (printTime == true) {
            fprintf(Display, "total time   = %.3f\n", cputime);
        }
#if DIMACS_PRINT
        fprintf(Display, "\n");
        gmp_fprintf(Display, "* DIMACS_ERRORS * \n");
        gmp_fprintf(Display, "err1 = %4.16Fe  [%40s]\n", err1.get_mpf_t(), "||Ax-b|| / (1+||b||_1) ");
        gmp_fprintf(Display, "err2 = %4.16Fe  [%40s]\n", err2.get_mpf_t(), "max(0, -lambda(x) / (1+||b||_1))");
        gmp_fprintf(Display, "err3 = %4.16Fe  [%40s]\n", err3.get_mpf_t(), "||A^Ty + z - c || / (1+||c||_1) ");
        gmp_fprintf(Display, "err4 = %4.16Fe  [%40s]\n", err4.get_mpf_t(), "max(0, -lambda(z) / (1+||c||_1))");
        gmp_fprintf(Display, "err5 = %4.16Fe  [%40s]\n", err5.get_mpf_t(), "(<c,x> - by) / (1 + |<c,x>| + |by|)");
        gmp_fprintf(Display, "err6 = %4.16Fe  [%40s]\n", err6.get_mpf_t(), "<x,z> / (1 + |<c,x>| + |by|)");
        fprintf(Display, "\n");
#endif
    }
    if (fpout) {
        fprintf(fpout, "\n");
        phase.display(fpout);
        fprintf(fpout, "   Iteration = %d\n", pIteration);
        gmp_fprintf(fpout, "          mu = %4.16Fe\n", mu.current.get_mpf_t());
        gmp_fprintf(fpout, "relative gap = %4.16Fe\n", relgap.get_mpf_t());
        gmp_fprintf(fpout, "         gap = %4.16Fe\n", gap.get_mpf_t());
        gmp_fprintf(fpout, "      digits = %4.16Fe\n", digits.get_mpf_t());

#if REVERSE_PRIMAL_DUAL
        mpf_class mtmp1 = -solveInfo.objValDual;
        mpf_class mtmp2 = -solveInfo.objValPrimal;
        gmp_fprintf(fpout, "objValPrimal = %10.16Fe\n", mtmp1.get_mpf_t());
        gmp_fprintf(fpout, "objValDual   = %10.16Fe\n", mtmp2.get_mpf_t());
        gmp_fprintf(fpout, "p.feas.error = %10.16Fe\n", currentRes.normDualMat.get_mpf_t());
        gmp_fprintf(fpout, "d.feas.error = %10.16Fe\n", currentRes.normPrimalVec.get_mpf_t());
        gmp_fprintf(fpout, "relative eps = %10.16Fe\n", Rlamch_gmp("E").get_mpf_t());
#else
        gmp_fprintf(fpout, "objValPrimal = %10.16Fe\n", solveInfo.objValPrimal.get_mpf_t());
        gmp_fprintf(fpout, "objValDual   = %10.16Fe\n", solveInfo.objValDual.get_mpf_t());
        gmp_fprintf(fpout, "p.feas.error = %10.16Fe\n", currentRes.normPrimalVec.get_mpf_t());
        gmp_fprintf(fpout, "d.feas.error = %10.16Fe\n", currentRes.normDualMat.get_mpf_t());
        gmp_fprintf(fpout, "relative eps = %10.16Fe\n", Rlamch_gmp("E").get_mpf_t());
#endif
        fprintf(fpout, "total time   = %.3f\n", cputime);
#if DIMACS_PRINT
        fprintf(fpout, "\n");
        gmp_fprintf(fpout, "* DIMACS_ERRORS * \n");
        gmp_fprintf(fpout, "err1 = %4.16Fe  [%40s]\n", err1.get_mpf_t(), "||Ax-b|| / (1+||b||_1) ");
        gmp_fprintf(fpout, "err2 = %4.16Fe  [%40s]\n", err2.get_mpf_t(), "max(0, -lambda(x) / (1+||b||_1))");
        gmp_fprintf(fpout, "err3 = %4.16Fe  [%40s]\n", err3.get_mpf_t(), "||A^Ty + z - c || / (1+||c||_1) ");
        gmp_fprintf(fpout, "err4 = %4.16Fe  [%40s]\n", err4.get_mpf_t(), "max(0, -lambda(z) / (1+||c||_1))");
        gmp_fprintf(fpout, "err5 = %4.16Fe  [%40s]\n", err5.get_mpf_t(), "(<c,x> - by) / (1 + |<c,x>| + |by|)");
        gmp_fprintf(fpout, "err6 = %4.16Fe  [%40s]\n", err6.get_mpf_t(), "<x,z> / (1 + |<c,x>| + |by|)");
        fprintf(fpout, "\n");
#endif

        fprintf(fpout, "\n\nParameters are\n");
        param.display(fpout);
        com.display(fpout);

#if 1
#if REVERSE_PRIMAL_DUAL
        fprintf(fpout, "xVec = \n");
        currentPt.yVec.display(fpout, -1.0, param.xPrint);
        fprintf(fpout, "xMat = \n");
        displayDenseLinarSpaceLast(currentPt.zMat, nBlock, blockStruct, blockType, blockNumber, param.XPrint, fpout);
        fprintf(fpout, "yMat = \n");
        displayDenseLinarSpaceLast(currentPt.xMat, nBlock, blockStruct, blockType, blockNumber, param.YPrint, fpout);
#else
        fprintf(fpout, "xMat = \n");
        displayDenseLinarSpaceLast(currentPt.xMat, nBlock, blockStruct, blockType, blockNumber, param.XPrint, fpout);
        fprintf(fpout, "yVec = \n");
        currentPt.yVec.display(fpout);
        fprintf(fpout, "zMat = \n");
        displayDenseLinarSpaceLast(currentPt.zMat, nBlock, blockStruct, blockType, blockNumber, param.YPrint, fpout);
#endif
#endif
    }
}

void IO::displayDenseLinarSpaceLast(DenseLinearSpace &aMat, int nBlock, int *blockStruct, int *blockType, int *blockNumber, const char *printFormat, FILE *fpout) {
    if (fpout == NULL) {
        return;
    }
    if (strcmp(printFormat, NO_P_FORMAT) == 0) {
        fprintf(fpout, "%s\n", NO_P_FORMAT);
        return;
    }
    fprintf(fpout, "{\n");
    for (int i = 0; i < nBlock; i++) {
        if (blockType[i] == 1) {
            int l = blockNumber[i];
            aMat.SDP_block[l].display(fpout, printFormat);
        } else if (blockType[i] == 2) {
            rError("io:: current version does not support SOCP");
            int l = blockNumber[i];
            aMat.SOCP_block[l].display(fpout);
        } else if (blockType[i] == 3) {
            fprintf(fpout, "{");
            for (int l = 0; l < blockStruct[i] - 1; ++l) {
                gmp_fprintf(fpout, printFormat, aMat.LP_block[blockNumber[i] + l].get_mpf_t());
                gmp_fprintf(fpout, ",");
            }
            if (blockStruct[i] > 0) {
                gmp_fprintf(fpout, printFormat, aMat.LP_block[blockNumber[i] + blockStruct[i] - 1].get_mpf_t());
                gmp_fprintf(fpout, "}\n");
            } else {
                gmp_fprintf(fpout, "  }\n");
            }
        } else {
            rError("io::displayDenseLinarSpaceLast not valid blockType");
        }
    }
    fprintf(fpout, "}\n");
}

} // namespace sdpa
