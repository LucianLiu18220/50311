/**
 *Submitted for verification at Etherscan.io on 2017-11-03
*/

pragma solidity^0.4.8;

contract BlockApps_Certificate_TO1103 {
    address public owner = msg.sender;
    string certificate;
    bool certIssued = false;

    function publishGraduatingClass(string cert) {
        if (msg.sender != owner || certIssued)
            throw;
        certIssued = true;
        certificate = cert;
    }
    function showCertificate() constant returns (string) {
        return certificate;
    }
}